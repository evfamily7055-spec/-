import streamlit as st
import pandas as pd
import re
import requests # Gemini API呼び出し用
import time # リトライ用
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import Counter
from itertools import combinations
import japanize_matplotlib # Matplotlibの日本語化
import numpy as np # --- 統計計算のために追加 ---
from scipy.stats import chi2_contingency # --- カイ二乗検定のために追加 ---
import io # --- 画像ダウンロード機能のために追加 ---
import base64 # --- HTMLレポートの画像埋込みのために追加 ---
from streamlit.components.v1 import html # --- KWIC表示用のhtmlコンポーネント ---

# --- 1. アプリの基本設定 ---
st.set_page_config(page_title="統計＋AI 統合アナライザー", layout="wide")
st.title("統計＋AI 統合テキストアナライザー 📊🤖")
st.write("Excelをアップロードし、テキスト列と分析軸（属性）を選択してください。統計分析とAIによる要約を同時に実行します。")

# --- 2. 形態素解析＆ストップワード設定 (キャッシュ) ---
@st.cache_resource # 形態素解析器は重いのでキャッシュ
def get_tokenizer():
    return Tokenizer()

BASE_STOPWORDS = set([
    'の', 'に', 'は', 'を', 'た', 'です', 'ます', 'が', 'で', 'も', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'する',
    'ない', 'こと', 'もの', 'これ', 'それ', 'あれ', 'よう', 'ため', '人', '中', '等', '思う', 'いう', 'なる', '日', '時',
    'くださる', 'いただく', 'しれる', 'くる', 'おる', 'れる', 'られる', 'せる', 'させる', 'できる', 'なる', 'やる', 'いく',
    '行う', '言う', '申し上げる', 'まいる', '見る', 'ここ', 'そこ', 'あそこ', 'こちら', 'そちら', 'あちら', 'この', 'その', 'あの'
])

# --- ▼ 修正点: AI分析の最大文字数制限を定義 ---
# APIの安定性を考慮し、AIに送信するテキストの最大文字数を設定（約100万文字）
MAX_AI_INPUT_CHARS = 1000000
# --- ▲ 修正完了 ▲ ---

@st.cache_data # テキストとTokenizerに変化がなければキャッシュ
def extract_words(text, _tokenizer): # _tokenizer引数はキャッシュのキーとして使う
    if not isinstance(text, str):
        return []
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        if token.surface.isdigit(): continue
        if token.base_form.isdigit(): continue
        part_of_speech = token.part_of_speech.split(',')[0]
        if part_of_speech in ['名詞', '動詞', '形容詞']:
            if len(token.base_form) > 1: # 文字数チェックのみ行う
                words.append(token.base_form)
    return words

# --- 3. Gemini AI 分析関数 (会話対応版) ---

# 1. シンプルな要約プロンプト (固定)
# --- ▼ 修正点: AIへの指示に「件数」を含めるよう変更 ---
SYSTEM_PROMPT_SIMPLE = """あなたは、テキストマイニングの専門家です。与えられたテキスト群を分析し、結果を詳細かつ分かりやすくマークダウン形式で出力してください。
{attributeInstruction}
## 1. 分析サマリー
(全体の傾向を簡潔に要約)
## 2. 主要なテーマ
(頻出するトピックや意見のカテゴリを3〜5個提示してください。**可能であれば、それぞれのテーマがおおよそ何件の意見に基づいているか（件数や割合）にも言及してください**。)
## 3. ポジティブな意見
(具体的な良い点を引用しつつリストアップ)
## 4. ネガティブな意見・課題
(具体的な不満や改善点を引用しつつリストアップ)
## 5. 少数だが注目すべき意見
(**件数は少ない（例: 1〜2件）かもしれないが、非常に重要、ユニーク、または示唆に富む意見があれば、ここに抽出してください**。)
{has_attribute}
## 7. 総評とネクストアクション
(分析から言えること、次に行うべきアクションを提案)
"""
# --- ▲ 修正完了 ▲ ---

# 2. 学術論文用プロンプト (固定)
# --- ▼ 修正点: AIへの指示に「件数」を含めるよう変更 ---
SYSTEM_PROMPT_ACADEMIC = """あなたは、テキストデータを分析する計量テキスト分析（テキストマイニング）の専門家です。
与えられたテキスト群を分析し、その結果を学術論文の「結果」および「考察」セクションに記述するのに適した、客観的かつフォーマルな文体で出力してください。
{attributeInstruction}
以下の構成に従って、マークダウン形式で記述してください。

## 1. 分析の概要 (Abstract)
(データ全体の主要な傾向や特筆すべき点を、客観的な要約として2〜3文で記述する)

## 2. 主要な知見 (Key Findings)
(データから抽出された主要なテーマやトピック、頻出する意見のカテゴリを3〜5点、箇条書きで提示する。**可能であれば、それぞれのテーマがおおよそ何件の意見に基づいているか（件数や割合）にも言及し**、具体的なテキスト断片を「」で引用して所見を補強する)

## 3. その他の注目すべき所見 (Other Notable Findings)
(**頻度は低い（例: 1〜2件）ものの、分析上見過ごすべきではない特異な意見、または将来の課題を示唆するような意見があれば、ここに記述する**)
{has_attribute}
## 5. 考察と今後の課題 (Discussion and Limitations)
(分析結果から導かれる考察や示唆を記述する。また、データから見られる潜在的な課題や、さらなる分析の方向性についても言及する)
"""
# --- ▲ 修正完了 ▲ ---


# 3. 会話用プロンプト (可変)
SYSTEM_PROMPT_CHAT = """あなたは、与えられたテキストデータ（コンテキスト）に関する質問に答える、優秀なデータアナリストです。
ユーザーからの質問に対し、提供されたコンテキスト情報に基づいて、簡潔かつ的確に回答してください。
コンテキストに含まれていない情報については、その旨を正直に伝えてください。
"""

def call_gemini_api(contents, system_instruction=None):
    try: apiKey = st.secrets["GEMINI_API_KEY"]
    except Exception: return "AI分析エラー: Streamlit CloudのSecretsに `GEMINI_API_KEY` が設定されていません。"
    if not apiKey: return "AI分析エラー: Streamlit CloudのSecretsに `GEMINI_API_KEY` が設定されていません。"

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"

    payload = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        response = None; delay = 1000
        for i in range(5):
            response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
            if response.status_code == 200: break
            elif response.status_code == 429 or response.status_code >= 500: time.sleep(delay / 1000); delay *= 2
            else: response.raise_for_status()
        if response.status_code != 200: return f"AI分析失敗 (Status: {response.status_code})"

        result = response.json()
        candidates = result.get('candidates')
        if not candidates: return "AI応答エラー: candidates is missing"
        content = candidates[0].get('content')
        if not content or not content.get('parts'): return "AI応答エラー: content or parts is missing"
        text = content['parts'][0].get('text', '')
        if not text: return "AIからの応答が空でした。"
        return text
    except Exception as e:
        if "403" in str(e): return "AI分析エラー: 403 Forbidden. APIキー/設定を確認してください。"
        return f"AI分析エラー: {e}"

# --- 4. KWIC（文脈検索）関数 ---
def generate_kwic_html(df, text_column, keyword, max_results=100):
    if not keyword: return "<p>キーワードを入力してください。</p>"
    try: search_pattern = keyword.replace('*', '.*'); kwic_pattern = re.compile(f'(.{{0,40}})({search_pattern})(.{{0,40}})', re.IGNORECASE)
    except re.error as e: return f"<p>キーワード検索エラー: {e}</p>"
    results = []
    for text in df[text_column].dropna():
        for match in kwic_pattern.finditer(text):
            if len(results) >= max_results: break
            left, center, right = match.groups()
            # HTMLエスケープはstreamlit.components.v1.htmlが自動で行うため不要
            html_row = f'<div style="margin-bottom: 10px; padding: 5px; border-bottom: 1px solid #eee; font-family: sans-serif;"><span style="text-align: right; display: inline-block; width: 45%; color: #555; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">...{left}</span><span style="background-color: yellow; font-weight: bold; padding: 2px 0;">{center}</span><span style="text-align: left; display: inline-block; width: 45%; color: #555; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{right}...</span></div>'
            results.append(html_row)
        if len(results) >= max_results: break
    if not results: return f"<p>キーワード「{keyword}」は見つかりませんでした。</p>"
    return f"<h4>「{keyword}」の検索結果 ({len(results)} 件)</h4><div style='height:400px; overflow-y:scroll; border:1px solid #eee; padding:10px;'>" + "".join(results) + "</div>"


# --- 5. 属性別 特徴語（カイ二乗検定）関数 ---
@st.cache_data
def calculate_characteristic_words(_df, attribute_col, text_col, _stopwords_set):
    results = {}
    try: unique_attrs = _df[attribute_col].dropna().unique()
    except KeyError: return {"error": "属性列が見つかりません。"}
    if len(unique_attrs) < 2: return {"error": "比較対象の属性が2つ未満です。"}
    all_words = set(word for sublist in _df['words'] for word in sublist if word not in _stopwords_set)
    total_docs = len(_df)
    for attr_value in unique_attrs:
        attr_df = _df[_df[attribute_col] == attr_value]; non_attr_df = _df[_df[attribute_col] != attr_value]
        total_docs_in_attr = len(attr_df); total_docs_not_in_attr = total_docs - total_docs_in_attr
        if total_docs_in_attr == 0 or total_docs_not_in_attr == 0: continue
        characteristic_words = []
        for word in all_words:
            a = sum(1 for words_list in attr_df['words'] if word in words_list); b = sum(1 for words_list in non_attr_df['words'] if word in words_list)
            c = total_docs_in_attr - a; d = total_docs_not_in_attr - b
            if (a+b) == 0 or (a+c) == 0 or (b+d) == 0 or (c+d) == 0: continue
            contingency_table = np.array([[a, b], [c, d]])
            try:
                with np.errstate(divide='ignore', invalid='ignore'): chi2, p, dof, expected = chi2_contingency(contingency_table)
                if p < 0.05 and a > expected[0, 0]: characteristic_words.append((word, p, chi2))
            except ValueError: continue
        characteristic_words.sort(key=lambda x: x[1]); results[attr_value] = characteristic_words[:20]
    return results

# --- 7. WordCloud生成関数 ---
def generate_wordcloud(_words_list, font_path, _stopwords_set):
    filtered_words = [word for word in _words_list if word not in _stopwords_set]
    if not filtered_words: return None, "表示する単語がありません（ストップワード除去後）"
    word_freq = Counter(filtered_words)
    try:
        if not font_path:
            fig_wc, ax = plt.subplots(figsize=(12, 6)); ax.text(0.5, 0.5, "日本語フォントが見つかりません", ha='center', va='center', fontsize=16); ax.axis('off')
            return fig_wc, "日本語フォントが見つかりませんでした。"
        else:
            wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path, max_words=100).generate_from_frequencies(word_freq)
            fig_wc, ax = plt.subplots(figsize=(12, 6)); ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            plt.close(fig_wc) # メモリ解放のためにCloseする
            return fig_wc, None
    except Exception as e: return None, f"WordCloud生成失敗: {e}"

# --- 8. 共起ネットワーク生成関数 ---
def generate_network(_words_df, font_path, _stopwords_set):
    co_occur_counter = Counter()
    for words in _words_df:
        unique_words = sorted(list(set(word for word in words if word not in _stopwords_set)))
        for w1, w2 in combinations(unique_words, 2): co_occur_counter[(w1, w2)] += 1
    top_pairs = co_occur_counter.most_common(50)
    if top_pairs:
        G = nx.Graph()
        for (w1, w2), weight in top_pairs: G.add_edge(w1, w2, weight=weight)
        fig_net, ax = plt.subplots(figsize=(14, 14)); pos = nx.spring_layout(G, k=0.8, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
        edge_weights = [d['weight'] * 0.2 for u,v,d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='gray')
        labels_kwargs = {'font_size': 10, 'font_family': 'IPAexGothic'} if font_path else {'font_size': 10}
        nx.draw_networkx_labels(G, pos, **labels_kwargs)
        ax.axis('off')
        plt.close(fig_net) # メモリ解放
        return fig_net, None
    return None, "共起ネットワーク生成不可（共起ペア不足）。"

# 単語頻度計算関数
def calculate_frequency(_words_list, _stopwords_set, top_n=50):
    filtered_words = [word for word in _words_list if word not in _stopwords_set]
    if not filtered_words: return pd.DataFrame(columns=['Rank', 'Word', 'Frequency'])
    word_freq = Counter(filtered_words).most_common(top_n)
    freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    freq_df['Rank'] = freq_df.index + 1
    return freq_df[['Rank', 'Word', 'Frequency']]

# HTMLレポート生成関数
def fig_to_base64_png(fig):
    if fig is None: return None
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def generate_html_report():
    html_parts = ["<!DOCTYPE html><html lang='ja'><head><meta charset='UTF-8'><title>テキスト分析レポート</title>"]
    html_parts.append("<style>body{font-family:sans-serif;margin:20px}h1,h2,h3{color:#333;border-bottom:1px solid #ccc;padding-bottom:5px}h2{margin-top:30px}.result-section{margin-bottom:30px;padding:15px;border:1px solid #eee;border-radius:5px;background-color:#f9f9f9}img{max-width:100%;height:auto;border:1px solid #ddd;margin-top:10px}table{border-collapse:collapse;width:100%;margin-top:10px}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}pre{background-color:#eee;padding:10px;border-radius:3px;white-space:pre-wrap;word-wrap:break-word}</style>")
    html_parts.append("</head><body><h1>テキスト分析レポート</h1>")
    if 'ai_result_simple' in st.session_state: html_parts.append(f"<div class='result-section'><h2>🤖 AI サマリー (簡易)</h2><pre>{st.session_state.ai_result_simple}</pre></div>")
    if 'fig_wc_display' in st.session_state and st.session_state.fig_wc_display:
        img_base64 = fig_to_base64_png(st.session_state.fig_wc_display);
        if img_base64: html_parts.append(f"<div class='result-section'><h2>☁️ WordCloud (全体)</h2><img src='{img_base64}' alt='WordCloud Overall'></div>")
    if 'overall_freq_df_display' in st.session_state and not st.session_state.overall_freq_df_display.empty:
        html_parts.append("<div class='result-section'><h2>📊 単語頻度ランキング (全体 Top 50)</h2>" + st.session_state.overall_freq_df_display.to_html(index=False) + "</div>")
    if 'fig_net_display' in st.session_state and st.session_state.fig_net_display:
        img_base64 = fig_to_base64_png(st.session_state.fig_net_display);
        if img_base64: html_parts.append(f"<div class='result-section'><h2>🕸️ 共起ネットワーク</h2><img src='{img_base64}' alt='Co-occurrence Network'></div>")
    if 'chi2_results_display' in st.session_state and st.session_state.chi2_results_display and "error" not in st.session_state.chi2_results_display:
        if st.session_state.attribute_columns: # 属性が選択されている場合のみ
            attr_col = st.session_state.attribute_columns[0]; html_parts.append(f"<div class='result-section'><h2>📈 属性別 特徴語 ({attr_col})</h2>")
            for attr_value, words in st.session_state.chi2_results_display.items():
                html_parts.append(f"<h3>{attr_value}</h3>");
                if words: html_parts.append("<ul>" + "".join(f"<li>{w} (p={p:.3f})</li>" for w, p, c in words) + "</ul>")
                else: html_parts.append("<p>特徴語なし</p>")
            html_parts.append("</div>")
    if 'ai_result_academic' in st.session_state: html_parts.append(f"<div class='result-section'><h2>📝 AI 学術論文</h2><pre>{st.session_state.ai_result_academic}</pre></div>")
    html_parts.append("</body></html>"); return "".join(html_parts)

# --- 9. メイン画面のUI ---
uploaded_file = st.file_uploader("1. Excelファイル (xlsx) をアップロード", type=["xlsx"])

def fig_to_bytes(fig):
    if fig is None: return None
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
    return buf.getvalue()

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if 'df_original' not in st.session_state or not st.session_state.df_original.equals(df):
             st.session_state.clear(); st.session_state.df_original = df
        st.subheader("読み込みデータ (先頭5件)"); st.dataframe(df.head())
        all_columns = df.columns.tolist()

        # --- 10. 分析設定 (サイドバー) ---
        with st.sidebar:
            st.header("⚙️ 分析設定")
            text_column = st.selectbox("分析したいテキスト列", all_columns, index=0)
            attribute_columns = st.multiselect("分析軸 (複数OK: 例: 年代, 性別)", all_columns)
            
            # --- ▼ 修正点: サンプル件数の指定を削除し、全件分析の注記を追加 ---
            # st.number_input(...) を削除
            st.info(f"AI分析は全件（最大{MAX_AI_INPUT_CHARS:,}文字）を対象とします。統計分析は常に全件が対象です。", icon="ℹ️")
            # --- ▲ 修正完了 ▲ ---

            st.markdown("---")
            run_button = st.button("分析を実行", type="primary", use_container_width=True)
            if 'df_analyzed' in st.session_state:
                st.markdown("---"); st.header("📊 レポート出力")
                try: html_content = generate_html_report(); st.download_button("HTMLレポートをダウンロード", html_content, "text_analysis_report.html", "text/html", use_container_width=True)
                except Exception as e: st.error(f"レポート生成エラー: {e}")

        # --- 11. 分析実行 (形態素解析のみ) ---
        if run_button:
            if not text_column: st.error("「分析したいテキスト列」を選択してください。")
            else:
                required_cols = [text_column] + attribute_columns
                if not all(col in df.columns for col in required_cols): st.error("選択された列が見つかりません。")
                else:
                    with st.spinner("ステップ1/1: 形態素解析を実行中..."):
                        df_analyzed = st.session_state.df_original.copy()
                        _tokenizer_instance = get_tokenizer()
                        df_analyzed['words'] = df_analyzed[text_column].apply(lambda x: extract_words(x, _tokenizer_instance))
                        st.session_state.df_analyzed = df_analyzed
                        st.session_state.text_column = text_column
                        st.session_state.attribute_columns = attribute_columns
                        
                        # --- ▼ 修正点: ai_sample_size をセッションから削除 ---
                        # st.session_state.ai_sample_size = ai_sample_size
                        # --- ▲ 修正完了 ▲ ---

                        st.session_state.pop('ai_result_simple', None); st.session_state.pop('ai_result_academic', None)
                        st.session_state.pop('fig_wc_display', None); st.session_state.pop('wc_error_display', None)
                        st.session_state.pop('fig_net_display', None); st.session_state.pop('net_error_display', None)
                        st.session_state.pop('chi2_results_display', None); st.session_state.pop('chi2_error_display', None)
                        st.session_state.pop('overall_freq_df_display', None)
                        st.session_state.pop('attribute_freq_dfs_display', None)
                        st.session_state.pop('dynamic_stopwords', None)
                        st.session_state.pop('chat_messages', None) # チャット履歴もクリア
                        st.success("形態素解析完了。結果タブで各分析を実行・表示します。")

        # --- 12. 結果表示 (オンデマンド + 動的ストップワード対応) ---
        if 'df_analyzed' in st.session_state:
            st.subheader("📊 分析結果")
            df_analyzed = st.session_state.df_analyzed
            text_column = st.session_state.text_column
            attribute_columns = st.session_state.attribute_columns

            if 'font_path' not in st.session_state:
                try: font_path = matplotlib.font_manager.findfont('IPAexGothic'); st.session_state.font_path = font_path
                except Exception: st.session_state.font_path = None
            font_path = st.session_state.font_path
            if font_path is None: st.warning("日本語フォント 'IPAexGothic' が見つかりませんでした。", icon="⚠️")

            st.markdown("---")
            st.subheader("⚙️ 表示用ストップワード設定")
            st.info("以下の単語リストは、下の各分析タブの表示にのみ影響します（AI分析は除く）。カンマ区切りで入力してください。")
            if 'dynamic_stopwords' not in st.session_state: st.session_state.dynamic_stopwords = ""
            dynamic_stopwords_input = st.text_area("追加の除外語 (カンマ区切り)", value=st.session_state.dynamic_stopwords, key="dynamic_sw_input")
            if dynamic_stopwords_input != st.session_state.dynamic_stopwords:
                st.session_state.pop('fig_wc_display', None); st.session_state.pop('wc_error_display', None)
                st.session_state.pop('fig_net_display', None); st.session_state.pop('net_error_display', None)
                st.session_state.pop('chi2_results_display', None); st.session_state.pop('chi2_error_display', None)
                st.session_state.pop('overall_freq_df_display', None)
                st.session_state.pop('attribute_freq_dfs_display', None)
                st.session_state.dynamic_stopwords = dynamic_stopwords_input # 新しい値を保存
            dynamic_sw_set = set(w.strip() for w in st.session_state.dynamic_stopwords.split(',') if w.strip())
            current_stopwords_set = BASE_STOPWORDS.union(dynamic_sw_set)
            st.markdown("---")

            tab_names = ["🤖 AI サマリー (簡易)", "☁️ WordCloud", "📊 単語頻度ランキング", "🕸️ 共起ネットワーク", "🔍 KWIC (文脈検索)", "📈 属性別 特徴語", "📝 AI 学術論文", "💬 AI チャット"]
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names) # タブ変数を8つに戻す

            # --- Tab 1: AI サマリー (簡易) ---
            with tab1:
                # (変更なし)
                if 'ai_result_simple' not in st.session_state:
                    with st.spinner("AIによる要約を生成中..."):
                        
                        # --- ▼ 修正点: サンプリング(sample)を廃止し、全件取得＋文字数制限に変更 ---
                        # sample_df = df_analyzed.sample(n=min(len(df_analyzed), 100))
                        
                        def format_for_ai(row):
                            text = row[text_column] or ''; attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                            return f"[{' | '.join(attrs)}] || {text}" if attrs else text
                        
                        # 全件を対象にする
                        ai_input_text_full = "\n".join(df_analyzed.apply(format_for_ai, axis=1))
                        
                        # 文字数制限をかける
                        if len(ai_input_text_full) > MAX_AI_INPUT_CHARS:
                            ai_input_text = ai_input_text_full[:MAX_AI_INPUT_CHARS]
                            st.warning(f"データが非常に大きいため、AI分析は先頭の約{MAX_AI_INPUT_CHARS:,}文字（全{len(df_analyzed):,}件中）を対象に実行されました。全件の厳密な統計は他のタブをご覧ください。", icon="⚠️")
                        else:
                            ai_input_text = ai_input_text_full
                            st.info(f"AI分析は全{len(df_analyzed):,}件を対象に実行されました。", icon="✅")
                        # --- ▲ 修正完了 ▲ ---

                        contents = [{"parts": [{"text": ai_input_text}]}]
                        has_attr = bool(attribute_columns)
                        has_attribute_str_s = "## 6. 属性別の傾向 (もしあれば)\n(属性ごとの特徴的な意見を比較)" if has_attr else ""
                        attr_instr_s = "データは「属性 || テキスト」の形式です。属性ごとの傾向や違いにも着目して分析してください。" if has_attr else ""
                        system_instr_s = SYSTEM_PROMPT_SIMPLE.format(attributeInstruction=attr_instr_s, has_attribute=has_attribute_str_s)
                        st.session_state.ai_result_simple = call_gemini_api(contents, system_instruction=system_instr_s)
                st.markdown(st.session_state.ai_result_simple)

            # --- Tab 2: WordCloud ---
            with tab2:
                # (変更なし、解説文のみ復元)
                st.subheader("全体のWordCloud")
                if 'fig_wc_display' not in st.session_state:
                    with st.spinner("WordCloudを生成中..."):
                        all_words_list = [word for sublist in df_analyzed['words'] for word in sublist]
                        fig_wc, wc_error = generate_wordcloud(all_words_list, font_path, current_stopwords_set)
                        st.session_state.fig_wc_display = fig_wc
                        st.session_state.wc_error_display = wc_error
                if st.session_state.fig_wc_display:
                    st.pyplot(st.session_state.fig_wc_display)
                    img_bytes = fig_to_bytes(st.session_state.fig_wc_display)
                    if img_bytes: st.download_button("この画像をダウンロード (PNG)", img_bytes, "wordcloud_overall.png", "image/png")
                else: st.warning(st.session_state.wc_error_display)

                # --- ▼ 修正点: 解説文を復元 ▼ ---
                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **形態素解析**: アップロードされたデータの指定テキスト列に対し、`Janome` ライブラリを用いて形態素解析を実行しました。
                        2.  **単語抽出**: 抽出する品詞を「名詞」「動詞」「形容詞」に限定しました。
                        3.  **ノイズ除去**: 一般的な助詞・助動詞（例: 「の」「です」）および、「表示用ストップワード設定」で指定された単語、数字、1文字の単語をストップワードとして分析から除外しました。
                        4.  **頻度集計**: 出現したすべての単語（基本形）の頻度をカウントしました。
                        5.  **可視化**: 上記の頻度データに基づき、`WordCloud` ライブラリを用いてワードクラウド（上位100語）を生成しました。
                        
                        #### 2. 論文記述例
                        > ...本研究では、[テキスト列名] の全体的な傾向を把握するため、形態素解析（ライブラリ: Janome）によりテキストを単語に分かち書きした。分析対象は名詞、動詞、形容詞の基本形に限定し、一般的すぎる助詞・助動詞や数字、および[ユーザー指定の単語]等をストップワードとして除外した。その上で、出現頻度上位100単語を対象にワードクラウドを生成した（図1参照）。
                        >
                        > 図1の結果から、[単語A]や[単語B]といった単語が特に大きく表示されており、[データ全体]においてこれらのトピックが頻繁に言及されていることが示唆された。
                    """)
                # --- ▲ 修正完了 ▲ ---

                st.markdown("---")
                st.subheader("属性別のWordCloud")
                if not attribute_columns: st.warning("属性別WordCloudを表示するには分析軸を選択してください。")
                else:
                    selected_attr_for_wc = st.selectbox("WordCloudの分析軸を選択", attribute_columns, 0, key="wc_attr_select")
                    if selected_attr_for_wc:
                        try: unique_values = sorted(df_analyzed[selected_attr_for_wc].dropna().unique())
                        except TypeError: unique_values = sorted(df_analyzed[selected_attr_for_wc].dropna().astype(str).unique())
                        st.info(f"「**{selected_attr_for_wc}**」の値ごとにWordCloudを生成します。")
                        for val in unique_values:
                            st.markdown(f"#### {selected_attr_for_wc} : **{val}**")
                            subset_df = df_analyzed[df_analyzed[selected_attr_for_wc] == val]
                            subset_words_list = [word for sublist in subset_df['words'] for word in sublist]
                            if not subset_words_list: st.info("単語なし"); continue
                            fig_subset_wc, wc_subset_error = generate_wordcloud(subset_words_list, font_path, current_stopwords_set)
                            if fig_subset_wc:
                                st.pyplot(fig_subset_wc)
                                img_bytes = fig_to_bytes(fig_subset_wc)
                                if img_bytes: st.download_button(f"「{val}」の画像をダウンロード", img_bytes, f"wordcloud_attr_{val}.png", "image/png")
                            else: st.warning(wc_subset_error)

            # --- Tab 3: 単語頻度ランキング ---
            with tab3:
                # (変更なし、解説文のみ復元)
                st.subheader("全体の単語頻度ランキング (Top 50)")
                if 'overall_freq_df_display' not in st.session_state:
                     with st.spinner("全体の単語頻度を計算中..."):
                        all_words_list = [word for sublist in df_analyzed['words'] for word in sublist]
                        overall_freq_df = calculate_frequency(all_words_list, current_stopwords_set)
                        st.session_state.overall_freq_df_display = overall_freq_df
                st.dataframe(st.session_state.overall_freq_df_display, use_container_width=True)

                # --- ▼ 修正点: 解説文を復元 ▼ ---
                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **単語抽出**: WordCloudと同様に、名詞・動詞・形容詞からストップワードと数字を除外した単語リストを使用しました。
                        2.  **頻度集計**: 全ドキュメントに出現したすべての単語（基本形）の頻度をカウントしました。
                        3.  **表示**: 頻度が高い順にソートし、上位50件を表形式で表示しました。
                        
                        #### 2. 論文記述例
                        > ...分析対象テキスト全体における主要な単語を特定するため、出現頻度分析を行った。形態素解析（前述）により抽出された単語（名詞、動詞、形容詞）の出現頻度を集計した結果、上位50単語は表Xの通りであった。
                        >
                        > 表Xより、[単語A] (N=[頻度])、[単語B] (N=[頻度]) が特に高頻度で出現しており、...
                    """)
                # --- ▲ 修正完了 ▲ ---

                st.markdown("---")
                st.subheader("属性別の単語頻度ランキング (Top 50)")
                if not attribute_columns: st.warning("属性別頻度を表示するには分析軸を選択してください。")
                else:
                    selected_attr_for_freq = st.selectbox("頻度ランキングの分析軸を選択", attribute_columns, 0, key="freq_attr_select")
                    if selected_attr_for_freq:
                        if ('attribute_freq_dfs_display' not in st.session_state or
                            st.session_state.get('attribute_freq_col_display') != selected_attr_for_freq or
                            st.session_state.get('attribute_freq_sw_display') != st.session_state.dynamic_stopwords):
                            with st.spinner(f"「{selected_attr_for_freq}」別の単語頻度を計算中..."):
                                st.session_state.attribute_freq_dfs_display = {}
                                try: unique_values = sorted(df_analyzed[selected_attr_for_freq].dropna().unique())
                                except TypeError: unique_values = sorted(df_analyzed[selected_attr_for_freq].dropna().astype(str).unique())
                                for val in unique_values:
                                    subset_df = df_analyzed[df_analyzed[selected_attr_for_freq] == val]
                                    subset_words_list = [word for sublist in subset_df['words'] for word in sublist]
                                    st.session_state.attribute_freq_dfs_display[val] = calculate_frequency(subset_words_list, current_stopwords_set)
                                st.session_state.attribute_freq_col_display = selected_attr_for_freq
                                st.session_state.attribute_freq_sw_display = st.session_state.dynamic_stopwords
                        attribute_freq_dfs = st.session_state.attribute_freq_dfs_display
                        st.info(f"「**{selected_attr_for_freq}**」の値ごとに単語頻度ランキング (Top 50) を表示します。")
                        for val, freq_df in attribute_freq_dfs.items():
                             with st.expander(f"属性: **{val}** のランキング"):
                                if freq_df.empty: st.info("単語なし")
                                else: st.dataframe(freq_df, use_container_width=True)

            # --- Tab 4: 共起ネットワーク ---
            with tab4:
                # (変更なし、解説文のみ復元)
                if 'fig_net_display' not in st.session_state:
                    with st.spinner("共起ネットワークを生成中..."):
                        fig_net, net_error = generate_network(df_analyzed['words'], font_path, current_stopwords_set)
                        st.session_state.fig_net_display = fig_net
                        st.session_state.net_error_display = net_error
                if st.session_state.fig_net_display:
                    st.pyplot(st.session_state.fig_net_display)
                    img_bytes = fig_to_bytes(st.session_state.fig_net_display)
                    if img_bytes: st.download_button("この画像をダウンロード (PNG)", img_bytes, "network.png", "image/png")
                else: st.warning(st.session_state.net_error_display)

                # --- ▼ 修正点: 解説文を復元 ▼ ---
                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **単語抽出**: WordCloudと同様に、名詞・動詞・形容詞からストップワードと数字を除外した単語リストを使用しました。
                        2.  **共起の定義**: 1つのドキュメント（Excelの1行）内で同時に出現した単語ペアを「共起」として定義しました。
                        3.  **頻度集計**: 全ドキュメントを対象に、共起する単語ペアの出現頻度を集計しました。
                        4.  **ネットワーク構築**: 共起頻度が高かった上位50ペアを抽出し、`NetworkX` ライブラリを用いてネットワークを構築しました。
                        5.  **可視化**: 単語をノード（点）、単語間の共起関係をエッジ（線）として描画しました。エッジの太さは共起頻度の高さ（関係の強さ）を反映しています（係数: 0.2）。レイアウトは `spring_layout` を使用しました。
                        
                        #### 2. 論文記述例
                        > ...次に、単語間の関連性を探索するため、共起ネットワーク分析を実施した。分析対象の単語（名詞、動詞、形容詞）が1ドキュメント（行）内で同時に出現した場合を「共起」と定義し、その頻度を集計した。共起頻度上位50ペアに基づきネットワーク（図2）を描画した。
                        >
                        > 図2より、[単語A]と[単語B]が強い共起関係（太いエッジ）にあることが確認された。また、[単語C]を中心として[単語D, E, F]がクラスターを形成しており、...といった文脈で語られていることが示唆された。
                    """)
                # --- ▲ 修正完了 ▲ ---

            # --- Tab 5: KWIC (文脈検索) ---
            with tab5:
                # (変更なし、解説文のみ復元)
                st.subheader("KWIC (文脈検索)")
                st.info("キーワードに `*` を含めるとワイルドカード検索が可能です (例: `顧客*`)。")
                kwic_keyword = st.text_input("文脈を検索したい単語を入力してください", key="kwic_input")
                if kwic_keyword:
                    kwic_html_content = generate_kwic_html(df_analyzed, text_column, kwic_keyword)
                    html(kwic_html_content, height=400, scrolling=True)

                # --- ▼ 修正点: 解説文を復元 ▼ ---
                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **キーワード検索**: 指定されたキーワード（`*` ワイルドカード利用可）に基づき、分析対象のテキスト列（原文）に対して正規表現検索を実行しました。
                        2.  **文脈抽出**: キーワードと一致した箇所の前後40文字を「文脈（コンテクスト）」として抽出し、一覧表示（コンコーダンス・ライン）しました。
                        
                        #### 2. 論文記述例
                        > ...共起ネットワーク分析で注目された[単語A]について、実際の文脈を詳細に確認するため、KWIC（KeyWord In Context）分析を行った。検索キーワード「[単語A]」で原文を検索した結果（表1）、...
                        >
                        > 表1（*KWICの結果を論文に引用*）
                        > ... [単語A]は、主に「...」といった文脈でポジティブに使用される一方、「...」というネガティブな文脈でも出現しており、...
                    """)
                # --- ▲ 修正完了 ▲ ---

            # --- Tab 6: 属性別 特徴語 ---
            with tab6:
                # (変更なし、解説文のみ復元)
                st.subheader("属性別 特徴語（カイ二乗検定）")
                if not attribute_columns: st.warning("この分析を行うには分析軸を選択してください。")
                else:
                    attr_col_for_chi2 = attribute_columns[0]
                    st.info(f"属性 「**{attr_col_for_chi2}**」 の値ごとに特徴的な単語を計算します。p値<0.05の有意な単語を表示。")
                    if ('chi2_results_display' not in st.session_state or
                        st.session_state.get('chi2_sw_display') != st.session_state.dynamic_stopwords):
                        with st.spinner(f"「{attr_col_for_chi2}」の特徴語を計算中..."):
                            chi2_results = calculate_characteristic_words(df_analyzed, attr_col_for_chi2, text_column, current_stopwords_set)
                            st.session_state.chi2_results_display = chi2_results
                            st.session_state.chi2_sw_display = st.session_state.dynamic_stopwords
                    chi2_results = st.session_state.chi2_results_display
                    if "error" in chi2_results: st.error(chi2_results["error"])
                    else:
                        if not chi2_results or all(not words for words in chi2_results.values()):
                            st.info(f"属性「{attr_col_for_chi2}」には統計的に有意な特徴語は見つかりませんでした。")
                        else:
                            cols = st.columns(len(chi2_results))
                            for i, (attr_value, words) in enumerate(chi2_results.items()):
                                with cols[i % len(cols)]:
                                    st.markdown(f"**{attr_value}** の特徴語 (Top 20)")
                                    if words:
                                        for word, p_value, chi2_val in words: st.write(f"- {word} (p={p_value:.3f})")
                                    else: st.info("特徴語なし")

                # --- ▼ 修正点: 解説文を復元 ▼ ---
                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **クロス集計**: 選択された属性（例: 「年代」）の各カテゴリ（例: 「20代」）と、分析対象の全単語（ストップワード等除外後）について、2x2のクロス集計表（分割表）を作成しました。
                            * a: 「20代」の文書で、単語Xを「含む」
                            * b: 「20代以外」の文書で、単語Xを「含む」
                            * c: 「20代」の文書で、単語Xを「含まない」
                            * d: 「20代以外」の文書で、単語Xを「含まない」
                        2.  **統計検定**: この集計表に対し、`Scipy` ライブラリを用いてカイ二乗検定（独立性の検定）を実行しました。
                        3.  **特徴語抽出**: 各カテゴリ（「20代」）において、(1) p値が 0.05 未満（統計的に有意）であり、(2) 実際の出現数(a)が期待値よりも高かった単語を、そのカテゴリの「特徴語」として抽出しました。
                        4.  **表示**: p値が低い順（＝より有意な順）にソートし、上位20件を表示しました。
                        
                        #### 2. 論文記述例
                        > ...属性[属性名]によるテキスト内容の差異を統計的に検証するため、特徴語抽出を行った。各属性カテゴリ（例: 「A群」）と全単語について2x2の分割表を作成し、カイ二乗検定（独立性の検定）を実施した。その結果、p値が0.05未満かつ残差が正であった単語を、各カテゴリの「特徴語」として抽出した（表2参照）。
                        >
                        > 表2の結果より、「A群」では[単語X, Y]が、「B群」では[単語Z]が特徴的に出現しており、...
                    """)
                # --- ▲ 修正完了 ▲ ---

            # --- Tab 7: AI 学術論文 ---
            with tab7:
                # (変更なし)
                st.subheader("AIによる学術論文風サマリー")
                if 'ai_result_academic' not in st.session_state:
                    with st.spinner("AIによる学術論文風の要約を生成中..."):

                        # --- ▼ 修正点: サンプリング(sample)を廃止し、全件取得＋文字数制限に変更 ---
                        # sample_df = df_analyzed.sample(n=min(len(df_analyzed), 100))
                        
                        def format_for_ai(row):
                            text = row[text_column] or ''; attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                            return f"[{' | '.join(attrs)}] || {text}" if attrs else text
                        
                        # 全件を対象にする
                        ai_input_text_full = "\n".join(df_analyzed.apply(format_for_ai, axis=1))
                        
                        # 文字数制限をかける
                        if len(ai_input_text_full) > MAX_AI_INPUT_CHARS:
                            ai_input_text = ai_input_text_full[:MAX_AI_INPUT_CHARS]
                            st.warning(f"データが非常に大きいため、AI分析は先頭の約{MAX_AI_INPUT_CHARS:,}文字（全{len(df_analyzed):,}件中）を対象に実行されました。全件の厳密な統計は他のタブをご覧ください。", icon="⚠️")
                        else:
                            ai_input_text = ai_input_text_full
                            st.info(f"AI分析は全{len(df_analyzed):,}件を対象に実行されました。", icon="✅")
                        # --- ▲ 修正完了 ▲ ---
                        
                        contents_acad = [{"parts": [{"text": ai_input_text}]}]
                        has_attr_a = bool(attribute_columns)
                        has_attribute_str_a = "## 4. 属性間の比較分析 (Comparative Analysis)\n(属性（カテゴリ）間で見られた顕著な差異や特徴的な傾向について、具体的に比較・記述する)" if has_attr_a else ""
                        attr_instr_a = "データは「属性 || テキスト」の形式です。属性ごとの傾向や違いにも着目して分析してください。" if has_attr_a else ""
                        system_instr_s = SYSTEM_PROMPT_ACADEMIC.format(attributeInstruction=attr_instr_a, has_attribute=has_attribute_str_a)
                        st.session_state.ai_result_academic = call_gemini_api(contents_acad, system_instruction=system_instr_a)
                st.markdown(st.session_state.ai_result_academic)
                
            # --- ▼ 修正点: AIチャットタブのコードをここに追加 ▼ ---
            with tab8:
                st.subheader("💬 AI チャット (データ分析)")
                st.info("分析されたデータ（全件または最大100万文字）について、AIに質問できます。")
                if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]): st.markdown(message["content"])
                if prompt := st.chat_input("AIに質問を入力してください (例: 主な課題は何ですか？)"):
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.spinner("AIが応答を生成中..."):
                        
                        # --- ▼ 修正点: サンプリング(sample)を廃止し、全件取得＋文字数制限に変更 ---
                        # sample_df_chat = df_analyzed.sample(n=min(len(df_analyzed), 100))
                        
                        def format_for_ai_chat(row):
                            text = row[text_column] or ''; attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                            return f"[{' | '.join(attrs)}] || {text}" if attrs else text
                        
                        # 全件を対象にする
                        context_text_full = "\n".join(df_analyzed.apply(format_for_ai_chat, axis=1))
                        
                        # 文字数制限をかける
                        if len(context_text_full) > MAX_AI_INPUT_CHARS:
                            context_text = context_text_full[:MAX_AI_INPUT_CHARS]
                            st.warning(f"データが非常に大きいため、AIのコンテキスト（参照データ）は先頭の約{MAX_AI_INPUT_CHARS:,}文字分となります。", icon="⚠️")
                        else:
                            context_text = context_text_full
                        # --- ▲ 修正完了 ▲ ---

                        api_contents = []
                        # 最初のユーザーターンにコンテキストを追加し、それ以降は単純な会話履歴とする
                        first_user_message = f"""以下のテキストデータ（コンテキスト）について質問があります。\n\n--- コンテキスト ---\n{context_text}\n\n--- 質問 ---\n{prompt}"""
                        is_first_turn = len(st.session_state.chat_messages) == 1
                        
                        # 既存の履歴を変換
                        for msg in st.session_state.chat_messages[:-1]:
                            api_contents.append({"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]})
                        # 新しいユーザーメッセージ（初回ならコンテキスト付き）を追加
                        api_contents.append({"role": "user", "parts": [{"text": first_user_message if is_first_turn else prompt}]})

                        response = call_gemini_api(api_contents, system_instruction=SYSTEM_PROMPT_CHAT)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"): st.markdown(response)
            # --- ▲ 修正完了 ▲ ---

    except Exception as e:
        st.error(f"ファイルの読み込みまたは分析中にエラーが発生しました: {e}")
        st.exception(e) # 詳細なエラーログを表示
