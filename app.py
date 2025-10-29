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
# --- ▼ 修正点: prince の import を削除 ▼ ---
# import prince 
# --- ▲ 修正完了 ▲ ---
import io # --- 画像ダウンロード機能のために追加 ---

# --- 1. アプリの基本設定 ---
st.set_page_config(page_title="統計＋AI 統合アナライザー", layout="wide")
st.title("統計＋AI 統合テキストアナライザー 📊🤖")
st.write("Excelをアップロードし、テキスト列と分析軸（属性）を選択してください。統計分析とAIによる要約を同時に実行します。")

# --- 2. 形態素解析＆ストップワード設定 (キャッシュ) ---
@st.cache_resource # 形態素解析器は重いのでキャッシュ
def get_tokenizer():
    return Tokenizer()

# --- ▼ 修正点: ストップワードを追加 ▼ ---
STOPWORDS = set([
    'の', 'に', 'は', 'を', 'た', 'です', 'ます', 'が', 'で', 'も', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'する',
    'ない', 'こと', 'もの', 'これ', 'それ', 'あれ', 'よう', 'ため', '人', '中', '等', '思う', 'いう', 'なる', '日', '時',
    'くださる', 'いただく', 'しれる', 'くる', 'おる', 'れる', 'られる', 'せる', 'させる', 'できる', 'なる', 'やる', 'いく',
    '行う', '言う', '申し上げる', 'まいる', '見る', 'ここ', 'そこ', 'あそこ', 'こちら', 'そちら', 'あちら', 'この', 'その', 'あの'
])
# --- ▲ 修正完了 ▲ ---

# テキストを単語リストに分割する関数
@st.cache_data # テキストとTokenizerに変化がなければキャッシュ
def extract_words(text, _tokenizer): # _tokenizer引数はキャッシュのキーとして使う
    if not isinstance(text, str):
        return []
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        
        # --- ▼ 修正点: 数字のみのトークンを除外するロジックを強化 ▼ ---
        # 表面形が数字のみかチェック
        if token.surface.isdigit():
            continue
        # 基本形が数字のみかチェック
        if token.base_form.isdigit():
            continue
        # --- ▲ 修正完了 ▲ ---
        
        part_of_speech = token.part_of_speech.split(',')[0]
        # 名詞、動詞、形容詞のみを抽出
        if part_of_speech in ['名詞', '動詞', '形容詞']:
            
            # ストップワードでなく、1文字以上なら追加
            if token.base_form not in STOPWORDS and len(token.base_form) > 1:
                words.append(token.base_form)
    return words

# --- 3. Gemini AI 分析関数 ---

# 1. シンプルな要約プロンプト (以前のバージョン)
SYSTEM_PROMPT_SIMPLE = """あなたは、テキストマイニングの専門家です。与えられたテキスト群を分析し、結果を詳細かつ分かりやすくマークダウン形式で出力してください。
{attributeInstruction}
## 1. 分析サマリー
(全体の傾向を簡潔に要約)
## 2. 主要なテーマ
(頻出するトピックや意見のカテゴリを3〜5個提示)
## 3. ポジティブな意見
(具体的な良い点を引用しつつリストアップ)
## 4. ネガティブな意見・課題
(具体的な不満や改善点を引用しつつリストアップ)
{has_attribute}
## 6. 総評とネクストアクション
(分析から言えること、次に行うべきアクションを提案)
"""

# 2. 学術論文用プロンプト
SYSTEM_PROMPT_ACADEMIC = """あなたは、テキストデータを分析する計量テキスト分析（テキストマイニング）の専門家です。
与えられたテキスト群を分析し、その結果を学術論文の「結果」および「考察」セクションに記述するのに適した、客観的かつフォーマルな文体で出力してください。
{attributeInstruction}
以下の構成に従って、マークダウン形式で記述してください。

## 1. 分析の概要 (Abstract)
(データ全体の主要な傾向や特筆すべき点を、客観的な要約として2〜3文で記述する)

## 2. 主要な知見 (Key Findings)
(データから抽出された主要なテーマやトピック、頻出する意見のカテゴリを3〜5点、箇条書きで提示する。可能であれば、具体的なテキスト断片を「」で引用し、所見を補強する)
{has_attribute}
## 4. 考察と今後の課題 (Discussion and Limitations)
(分析結果から導かれる考察や示唆を記述する。また、データから見られる潜在的な課題や、さらなる分析の方向性についても言及する)
"""


@st.cache_data # AI分析結果もキャッシュ
def call_gemini_api(text_data, has_attribute, prompt_type='simple'):
    try:
        apiKey = st.secrets["GEMINI_API_KEY"]
        if not apiKey:
            return "AI分析エラー: Streamlit CloudのSecretsに `GEMINI_API_KEY` が設定されていません。"
    except Exception:
         return "AI分析エラー: Streamlit CloudのSecretsに `GEMINI_API_KEY` が設定されていません。アプリ右下の 'Manage app' > 'Settings' > 'Secrets' を確認してください。"

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"

    attributeInstruction = ("データは「属性 || テキスト」の形式です。属性ごとの傾向や違いにも着目して分析してください。"
                            if has_attribute
                            else "")
    
    if prompt_type == 'academic':
        has_attribute_str = "## 3. 属性間の比較分析 (Comparative Analysis)\n(属性（カテゴリ）間で見られた顕著な差異や特徴的な傾向について、具体的に比較・記述する)" if has_attribute else ""
        systemPrompt = SYSTEM_PROMPT_ACADEMIC.format(attributeInstruction=attributeInstruction, has_attribute=has_attribute_str)
    else: # 'simple' (default)
        has_attribute_str = "## 5. 属性別の傾向 (もしあれば)\n(属性ごとの特徴的な意見を比較)" if has_attribute else ""
        systemPrompt = SYSTEM_PROMPT_SIMPLE.format(attributeInstruction=attributeInstruction, has_attribute=has_attribute_str)
    
    payload = {
        "systemInstruction": {"parts": [{"text": systemPrompt}]},
        "contents": [{"parts": [{"text": text_data}]}]
    }

    try:
        response = None
        delay = 1000
        for i in range(5): # Exponential backoff retry
            response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                break
            elif response.status_code == 429 or response.status_code >= 500:
                st.warning(f"AI APIがビジーです。リトライします... ({i+1}/5)", icon="⏳")
                time.sleep(delay / 1000)
                delay *= 2
            else:
                response.raise_for_status() # 他のエラー
        
        if response.status_code != 200:
             return f"AI分析に失敗しました。リトライ後もエラーが解消しませんでした。 (Status: {response.status_code})"

        result = response.json()
        
        # --- ▼ 修正点: AIからの応答がない場合のメッセージを改善 ▼ ---
        candidates = result.get('candidates')
        if not candidates:
            st.error("AIからの応答がありませんでした (candidates is missing)。APIの制限に達したか、設定に問題がある可能性があります。", icon="🚨")
            return "AIからの応答がありませんでした。"
            
        content = candidates[0].get('content')
        if not content or not content.get('parts'):
            st.error("AIからの応答形式が不正です (content or parts is missing)。", icon="🚨")
            return "AIからの応答形式が不正です。"
            
        text = content['parts'][0].get('text', '')
        if not text:
             st.warning("AIからの応答が空でした。プロンプトやデータ内容を見直してください。", icon="⚠️")
             return "AIからの応答が空でした。"
        return text
        # --- ▲ 修正完了 ▲ ---

    except Exception as e:
        if "403" in str(e):
            return "AI分析中にエラーが発生しました: 403 Forbidden. APIキーが正しいか、Google CloudでGemini APIが有効になっているか確認してください。"
        return f"AI分析中にエラーが発生しました: {e}"

# --- 4. KWIC（文脈検索）関数 ---
def generate_kwic_html(df, text_column, keyword, max_results=100):
    if not keyword:
        return "<p>キーワードを入力してください。</p>"
    
    # 大文字小文字を区別しない正規表現
    try:
        # ワイルドカードを考慮
        search_pattern = keyword.replace('*', '.*')
        kwic_pattern = re.compile(f'(.{{0,40}})({search_pattern})(.{{0,40}})', re.IGNORECASE)
    except re.error as e:
        return f"<p>キーワード検索エラー: {e}</p>"

    results = []
    for text in df[text_column].dropna():
        for match in kwic_pattern.finditer(text):
            if len(results) >= max_results:
                break
            left, center, right = match.groups()
            html_row = f"""
            <div style="margin-bottom: 10px; padding: 5px; border-bottom: 1px solid #eee; font-family: sans-serif;">
                <span style="text-align: right; display: inline-block; width: 45%; color: #555; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">...{left}</span>
                <span style="background-color: yellow; font-weight: bold; padding: 2px 0;">{center}</span>
                <span style="text-align: left; display: inline-block; width: 45%; color: #555; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{right}...</span>
            </div>
            """
            results.append(html_row)
        if len(results) >= max_results:
            break
            
    if not results:
        return f"<p>キーワード「{keyword}」は見つかりませんでした。</p>"
        
    return f"<h4>「{keyword}」の検索結果 ({len(results)} 件)</h4>" + "".join(results)

# --- 5. 属性別 特徴語（カイ二乗検定）関数 ---
@st.cache_data
def calculate_characteristic_words(_df, attribute_col, text_col): # _df を受け取る
    results = {}
    
    # 属性のユニークな値を取得
    try:
        unique_attrs = _df[attribute_col].dropna().unique()
    except KeyError:
        return {"error": "属性列が見つかりません。"}
        
    if len(unique_attrs) < 2:
        return {"error": "比較対象の属性が2つ未満です。"}
    
    # 全体の語彙（単語リスト）を作成
    all_words = set(word for sublist in _df['words'] for word in sublist)
    total_docs = len(_df)
    
    for attr_value in unique_attrs:
        
        attr_df = _df[_df[attribute_col] == attr_value]
        non_attr_df = _df[_df[attribute_col] != attr_value]
        
        total_docs_in_attr = len(attr_df)
        total_docs_not_in_attr = total_docs - total_docs_in_attr
        
        if total_docs_in_attr == 0 or total_docs_not_in_attr == 0:
            continue
            
        characteristic_words = []
        
        for word in all_words:
            # カイ二乗検定のための分割表を作成
            # a: 属性Aに含まれ、単語Xも含む文書数
            a = sum(1 for words_list in attr_df['words'] if word in words_list)
            # b: 属性A以外に含まれ、単語Xを含む文書数
            b = sum(1 for words_list in non_attr_df['words'] if word in words_list)
            
            # c: 属性Aに含まれ、単語Xを含まない文書数
            c = total_docs_in_attr - a
            # d: 属性A以外に含まれ、単語Xを含まない文書数
            d = total_docs_not_in_attr - b
            
            # 期待値が0の場合を避ける (chi2_contingencyのValueError対策)
            if (a+b) == 0 or (a+c) == 0 or (b+d) == 0 or (c+d) == 0:
                continue

            contingency_table = np.array([[a, b], [c, d]])
            
            try:
                # 警告を無視する設定
                with np.errstate(divide='ignore', invalid='ignore'):
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                
                # 単語がこの属性で「より多く」使われているかチェック (期待値と比較)
                # p値が有意水準以下 (例: 0.05) かつ、実際の出現頻度が期待値より高い場合
                if p < 0.05 and a > expected[0, 0]:
                    characteristic_words.append((word, p, chi2))
            except ValueError:
                continue # 期待値に0が含まれる場合など

        # p値でソート (p値が小さいほど統計的に有意)
        characteristic_words.sort(key=lambda x: x[1])
        results[attr_value] = characteristic_words[:20] # 上位20件
        
    return results

# --- ▼ 修正点: 対応分析 関数 を削除 ▼ ---
# def run_correspondence_analysis(...):
# --- ▲ 修正完了 ▲ ---

# --- 7. WordCloud生成関数 (キャッシュ) ---
@st.cache_data
def generate_wordcloud(_words_list, font_path):
    if not _words_list:
        return None, "単語がありません"
    
    word_freq = Counter(_words_list)
    try:
        if not font_path:
            fig_wc, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "日本語フォントが見つかりません", ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig_wc, "日本語フォントが見つかりませんでした。"
        else:
            # --- ▼ 修正点: max_words を 100 に設定 ▼ ---
            wc = WordCloud(width=800, height=400, background_color='white',
                           font_path=font_path, max_words=100).generate_from_frequencies(word_freq)
            # --- ▲ 修正完了 ▲ ---
            fig_wc, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            return fig_wc, None
    except Exception as e:
        return None, f"WordCloudの生成に失敗しました。エラー: {e}"

# --- 8. 共起ネットワーク生成関数 (キャッシュ) ---
@st.cache_data
def generate_network(_words_df, font_path):
    co_occur_counter = Counter()
    for words in _words_df: # df['words'] を受け取る
        # --- ▼ 修正点: 数字を除外 (extract_wordsで行うのでここでは不要かも) ▼ ---
        # words_filtered = [w for w in words if not w.isdigit()] # 一応残す
        # unique_words = sorted(list(set(words_filtered)))
        unique_words = sorted(list(set(words))) # extract_wordsを信じる
        # --- ▲ 修正完了 ▲ ---
        for w1, w2 in combinations(unique_words, 2):
            co_occur_counter[(w1, w2)] += 1
    
    top_pairs = co_occur_counter.most_common(50)
    
    if top_pairs:
        G = nx.Graph()
        for (w1, w2), weight in top_pairs:
            G.add_edge(w1, w2, weight=weight)
        
        fig_net, ax = plt.subplots(figsize=(14, 14))
        pos = nx.spring_layout(G, k=0.8, iterations=50) # レイアウト計算回数を増やして安定化
        
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
        
        # エッジの太さの係数を 0.2 に
        edge_weights = [d['weight'] * 0.2 for u,v,d in G.edges(data=True)] # 線の太さ調整
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='gray')
        
        # 日本語フォント指定
        if font_path:
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='IPAexGothic')
        else:
            nx.draw_networkx_labels(G, pos, font_size=10) # フォントがない場合はデフォルトで

        ax.axis('off')
        return fig_net, None
    return None, "共起ネットワークを生成できませんでした（共起ペア不足）。"

# 単語頻度計算関数
@st.cache_data
def calculate_frequency(_words_list, top_n=50):
    if not _words_list:
        return pd.DataFrame(columns=['Rank', 'Word', 'Frequency'])
    word_freq = Counter(_words_list).most_common(top_n)
    freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    freq_df['Rank'] = freq_df.index + 1
    return freq_df[['Rank', 'Word', 'Frequency']]


# --- 9. メイン画面のUI ---
uploaded_file = st.file_uploader("1. Excelファイル (xlsx) をアップロード", type=["xlsx"])

# グラフをbytesに変換するヘルパー関数
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if 'df_original' not in st.session_state or not st.session_state.df_original.equals(df):
             st.session_state.clear() # ファイルが変わったらキャッシュをクリア
             st.session_state.df_original = df
        
        st.subheader("読み込みデータ (先頭5件)")
        st.dataframe(df.head())
        
        all_columns = df.columns.tolist()
        
        # --- 10. 分析設定 (サイドバー) ---
        with st.sidebar:
            st.header("⚙️ 分析設定")
            text_column = st.selectbox("分析したいテキスト列", all_columns, index=0)
            
            attribute_columns = st.multiselect("分析軸 (複数OK: 例: 年代, 性別)", all_columns)
            
            st.markdown("---")
            run_button = st.button("分析を実行", type="primary", use_container_width=True)

        # --- 11. 分析実行 (形態素解析のみ) ---
        if run_button:
            if not text_column:
                st.error("「分析したいテキスト列」を選択してください。")
            else:
                required_cols = [text_column] + attribute_columns
                if not all(col in df.columns for col in required_cols):
                    st.error(f"選択された列の一部がExcel内に見つかりません。列名を確認してください。")
                else:
                    # --- 11a. 統計分析 (形態素解析) ---
                    with st.spinner("ステップ1/1: 形態素解析を実行中... (全件処理)"):
                        df_analyzed = st.session_state.df_original.copy()
                        _tokenizer_instance = get_tokenizer() # ダミーでインスタンスを渡す
                        df_analyzed['words'] = df_analyzed[text_column].apply(lambda x: extract_words(x, _tokenizer_instance))
                        
                        # --- 11b. 必須情報をセッションに保存 ---
                        st.session_state.df_analyzed = df_analyzed
                        st.session_state.text_column = text_column
                        st.session_state.attribute_columns = attribute_columns
                        
                        # 古い計算結果をクリア
                        st.session_state.pop('ai_result_simple', None)
                        st.session_state.pop('ai_result_academic', None)
                        st.session_state.pop('fig_wc', None)
                        st.session_state.pop('wc_error', None)
                        st.session_state.pop('fig_net', None)
                        st.session_state.pop('net_error', None)
                        st.session_state.pop('chi2_results', None)
                        # --- ▼ 修正点: 対応分析関連のキャッシュクリアを削除 ▼ ---
                        # st.session_state.pop('fig_ca', None)
                        # st.session_state.pop('ca_error', None)
                        # --- ▲ 修正完了 ▲ ---
                        st.session_state.pop('overall_freq_df', None)
                        st.session_state.pop('attribute_freq_dfs', None)

                        st.success("形態素解析が完了しました。結果タブをクリックして各分析を実行してください。")

        # --- 12. 結果表示 (オンデマンド計算) ---
        if 'df_analyzed' in st.session_state:
            st.subheader("📊 分析結果")
            
            # 必要な変数をセッションから取得
            df_analyzed = st.session_state.df_analyzed
            text_column = st.session_state.text_column
            attribute_columns = st.session_state.attribute_columns
            
            # フォントパス（全グラフで共有）
            if 'font_path' not in st.session_state:
                try:
                    font_path = matplotlib.font_manager.findfont('IPAexGothic')
                    st.session_state.font_path = font_path
                except Exception:
                    st.session_state.font_path = None
            font_path = st.session_state.font_path
            if font_path is None:
                 st.warning("日本語フォント 'IPAexGothic' が見つかりませんでした。グラフの日本語が文字化けする可能性があります。", icon="⚠️")

            # --- ▼ 修正点: タブ名と数を変更 (対応分析を削除) ▼ ---
            tab_names = ["🤖 AI サマリー (簡易)", "☁️ WordCloud", "📊 単語頻度ランキング", "🕸️ 共起ネットワーク", "🔍 KWIC (文脈検索)", "📈 属性別 特徴語", "📝 AI 学術論文"]
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names) # タブ変数を7つに
            # --- ▲ 修正完了 ▲ ---
            
            # --- Tab 1: AI サマリー (簡易) ---
            with tab1:
                if 'ai_result_simple' not in st.session_state:
                    with st.spinner("AIによる要約を生成中..."):
                        sample_df = df_analyzed.sample(n=min(len(df_analyzed), 100))
                        
                        def format_for_ai(row):
                            text = row[text_column] or ''
                            attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                            if attrs:
                                return f"[{' | '.join(attrs)}] || {text}"
                            return text
                            
                        ai_input_text = "\n".join(sample_df.apply(format_for_ai, axis=1))
                        st.session_state.ai_result_simple = call_gemini_api(ai_input_text, has_attribute=bool(attribute_columns), prompt_type='simple')
                
                st.markdown(st.session_state.ai_result_simple)
            
            # --- Tab 2: WordCloud ---
            with tab2:
                st.subheader("全体のWordCloud")
                if 'fig_wc' not in st.session_state:
                    with st.spinner("WordCloudを計算中..."):
                        all_words_list = [word for sublist in df_analyzed['words'] for word in sublist]
                        fig_wc, wc_error = generate_wordcloud(all_words_list, font_path)
                        st.session_state.fig_wc = fig_wc
                        st.session_state.wc_error = wc_error
                
                if st.session_state.fig_wc:
                    st.pyplot(st.session_state.fig_wc)
                    st.download_button(
                        label="この画像をダウンロード (PNG)",
                        data=fig_to_bytes(st.session_state.fig_wc),
                        file_name="wordcloud_overall.png",
                        mime="image/png"
                    )
                else:
                    st.warning(st.session_state.wc_error)
                
                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **形態素解析**: アップロードされたデータの指定テキスト列に対し、`Janome` ライブラリを用いて形態素解析を実行しました。
                        2.  **単語抽出**: 抽出する品詞を「名詞」「動詞」「形容詞」に限定しました。
                        3.  **ノイズ除去**: 一般的な助詞・助動詞（例: 「の」「です」）および、ユーザー指定の単語（例: 「くる」「いただく」「行う」「見る」等）、数字、1文字の単語をストップワードとして分析から除外しました。
                        4.  **頻度集計**: 出現したすべての単語（基本形）の頻度をカウントしました。
                        5.  **可視化**: 上記の頻度データに基づき、`WordCloud` ライブラリを用いてワードクラウド（上位100語）を生成しました。
                        
                        #### 2. 論文記述例
                        > ...本研究では、[テキスト列名] の全体的な傾向を把握するため、形態素解析（ライブラリ: Janome）によりテキストを単語に分かち書きした。分析対象は名詞、動詞、形容詞の基本形に限定し、一般的すぎる助詞・助動詞や数字、および[ユーザー指定の単語]等をストップワードとして除外した。その上で、出現頻度上位100単語を対象にワードクラウドを生成した（図1参照）。
                        >
                        > 図1の結果から、[単語A]や[単語B]といった単語が特に大きく表示されており、[データ全体]においてこれらのトピックが頻繁に言及されていることが示唆された。
                    """)
                
                st.markdown("---")
                st.subheader("属性別のWordCloud")
                
                if not attribute_columns:
                    st.warning("属性別WordCloudを表示するには、サイドバーで「分析軸」を1つ以上選択してください。")
                else:
                    selected_attr_for_wc = st.selectbox(
                        "WordCloudの分析軸を選択", 
                        options=attribute_columns, 
                        index=0, 
                        key="wc_attr_select"
                    )
                    
                    if selected_attr_for_wc:
                        try:
                            unique_values = sorted(df_analyzed[selected_attr_for_wc].dropna().unique())
                        except TypeError: # 数値と文字列が混在している場合
                            unique_values = sorted(df_analyzed[selected_attr_for_wc].dropna().astype(str).unique())

                        st.info(f"「**{selected_attr_for_wc}**」の値ごとにWordCloudを生成します。")
                        
                        for val in unique_values:
                            st.markdown(f"#### {selected_attr_for_wc} : **{val}**")
                            subset_df = df_analyzed[df_analyzed[selected_attr_for_wc] == val]
                            subset_words_list = [word for sublist in subset_df['words'] for word in sublist]
                            
                            if not subset_words_list:
                                st.info("このカテゴリには表示する単語がありません。")
                                continue
                            
                            # 属性別WordCloudはキャッシュしない（インタラクティブ性重視）
                            fig_subset_wc, ax = plt.subplots(figsize=(10, 5))
                            if not font_path:
                                ax.text(0.5, 0.5, "日本語フォントが見つかりません", ha='center', va='center', fontsize=14)
                                ax.axis('off')
                            else:
                                try:
                                    # --- ▼ 修正点: max_words を 100 に設定 ▼ ---
                                    wc = WordCloud(width=600, height=300, background_color='white',
                                                   font_path=font_path, max_words=100).generate_from_frequencies(Counter(subset_words_list))
                                    # --- ▲ 修正完了 ▲ ---
                                    ax.imshow(wc, interpolation='bilinear')
                                    ax.axis('off')
                                except Exception as e:
                                    st.error(f"WordCloud(属性:{val})の生成に失敗: {e}")
                            
                            st.pyplot(fig_subset_wc)
                            st.download_button(
                                label=f"「{val}」の画像をダウンロード (PNG)",
                                data=fig_to_bytes(fig_subset_wc),
                                file_name=f"wordcloud_attr_{val}.png",
                                mime="image/png"
                            )

            # --- Tab 3: 単語頻度ランキング ---
            with tab3:
                st.subheader("全体の単語頻度ランキング (Top 50)")
                if 'overall_freq_df' not in st.session_state:
                    with st.spinner("全体の単語頻度を計算中..."):
                        all_words_list = [word for sublist in df_analyzed['words'] for word in sublist]
                        st.session_state.overall_freq_df = calculate_frequency(all_words_list)
                
                st.dataframe(st.session_state.overall_freq_df, use_container_width=True)
                
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

                st.markdown("---")
                st.subheader("属性別の単語頻度ランキング (Top 50)")

                if not attribute_columns:
                    st.warning("属性別の頻度ランキングを表示するには、サイドバーで「分析軸」を1つ以上選択してください。")
                else:
                    selected_attr_for_freq = st.selectbox(
                        "頻度ランキングの分析軸を選択", 
                        options=attribute_columns, 
                        index=0, 
                        key="freq_attr_select"
                    )
                    
                    if selected_attr_for_freq:
                        if 'attribute_freq_dfs' not in st.session_state or st.session_state.get('attribute_freq_col') != selected_attr_for_freq:
                            with st.spinner(f"「{selected_attr_for_freq}」別の単語頻度を計算中..."):
                                st.session_state.attribute_freq_dfs = {}
                                try:
                                    unique_values = sorted(df_analyzed[selected_attr_for_freq].dropna().unique())
                                except TypeError:
                                    unique_values = sorted(df_analyzed[selected_attr_for_freq].dropna().astype(str).unique())
                                
                                for val in unique_values:
                                    subset_df = df_analyzed[df_analyzed[selected_attr_for_freq] == val]
                                    subset_words_list = [word for sublist in subset_df['words'] for word in sublist]
                                    st.session_state.attribute_freq_dfs[val] = calculate_frequency(subset_words_list)
                                st.session_state.attribute_freq_col = selected_attr_for_freq # 計算した属性を記録
                        
                        attribute_freq_dfs = st.session_state.attribute_freq_dfs
                        
                        st.info(f"「**{selected_attr_for_freq}**」の値ごとに単語頻度ランキング (Top 50) を表示します。")
                        
                        # --- ▼ 修正点: マトリクス表示からExpander表示に変更 ▼ ---
                        for val, freq_df in attribute_freq_dfs.items():
                             with st.expander(f"属性: **{val}** のランキング"):
                                if freq_df.empty:
                                    st.info("単語なし")
                                else:
                                    st.dataframe(freq_df, use_container_width=True) # コンテナ幅いっぱいに表示
                        # --- ▲ 修正完了 ▲ ---
            
            # --- Tab 4: 共起ネットワーク ---
            with tab4: # 以前のtab3がtab4に
                if 'fig_net' not in st.session_state:
                     with st.spinner("共起ネットワークを計算中..."):
                        fig_net, net_error = generate_network(df_analyzed['words'], font_path)
                        st.session_state.fig_net = fig_net
                        st.session_state.net_error = net_error

                if st.session_state.fig_net:
                    st.pyplot(st.session_state.fig_net)
                    st.download_button(
                        label="この画像をダウンロード (PNG)",
                        data=fig_to_bytes(st.session_state.fig_net),
                        file_name="network.png",
                        mime="image/png"
                    )
                else:
                    st.warning(st.session_state.net_error)
                
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
            
            # --- Tab 5: KWIC (文脈検索) ---
            with tab5: # 以前のtab4がtab5に
                st.subheader("KWIC (文脈検索)")
                st.info("キーワードに `*` を含めるとワイルドカード検索が可能です (例: `顧客*`)。")
                kwic_keyword = st.text_input("文脈を検索したい単語を入力してください", key="kwic_input")
                if kwic_keyword:
                    # KWICは毎回再計算（キャッシュしない）
                    kwic_html = generate_kwic_html(df_analyzed, text_column, kwic_keyword)
                    st.markdown(kwic_html, unsafe_allow_html=True) 
                
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
            
            # --- Tab 6: 属性別 特徴語 ---
            with tab6: # 以前のtab5がtab6に
                st.subheader("属性別 特徴語（カイ二乗検定）")
                if not attribute_columns:
                    st.warning("この分析を行うには、サイドバーで「分析軸」を1つ以上選択してください。")
                else:
                    attr_col_for_chi2 = attribute_columns[0] 
                    st.info(f"属性 「**{attr_col_for_chi2}**」 の値ごとに特徴的な単語を計算します。p値が0.05未満の有意な単語を表示します。")
                    
                    if 'chi2_results' not in st.session_state:
                        with st.spinner(f"「{attr_col_for_chi2}」の特徴語を計算中..."):
                            st.session_state.chi2_results = calculate_characteristic_words(df_analyzed, attr_col_for_chi2, text_column)
                    
                    chi2_results = st.session_state.chi2_results
                    if "error" in chi2_results:
                        st.error(chi2_results["error"])
                    else:
                        if not chi2_results or all(not words for words in chi2_results.values()): # 全カテゴリで結果がない場合も考慮
                            st.info(f"属性「{attr_col_for_chi2}」には、統計的に有意な特徴語は見つかりませんでした。")
                        else:
                            # st.columns を使用
                            cols = st.columns(len(chi2_results))
                            for i, (attr_value, words) in enumerate(chi2_results.items()):
                                with cols[i % len(cols)]: # インデックスが列数を超えたら折り返す
                                    st.markdown(f"**{attr_value}** の特徴語 (Top 20)")
                                    if words:
                                        for word, p_value, chi2_val in words:
                                            st.write(f"- {word} (p={p_value:.3f})")
                                    else:
                                        st.info("特徴語なし")
                
                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **クロス集計**: 選択された属性（例: 「年代」）の各カテゴリ（例: 「20代」）と、分析対象の全単語について、2x2のクロス集計表（分割表）を作成しました。
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
            
            # --- ▼ 修正点: Tab 7 (旧対応分析) を削除し、Tab 8 を Tab 7 に変更 ▼ ---
            # with tab7: # 旧対応分析タブ削除
            
            with tab7: # 旧 Tab 8 (AI 学術論文) を Tab 7 に
            # --- ▲ 修正完了 ▲ ---
                st.subheader("AIによる学術論文風サマリー")
                if 'ai_result_academic' not in st.session_state:
                    with st.spinner("AIによる学術論文風の要約を生成中..."):
                        sample_df = df_analyzed.sample(n=min(len(df_analyzed), 100))
                        
                        def format_for_ai(row):
                            text = row[text_column] or ''
                            attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                            if attrs:
                                return f"[{' | '.join(attrs)}] || {text}"
                            return text
                            
                        ai_input_text = "\n".join(sample_df.apply(format_for_ai, axis=1))
                        # prompt_type='academic' で呼び出し
                        st.session_state.ai_result_academic = call_gemini_api(ai_input_text, has_attribute=bool(attribute_columns), prompt_type='academic')
                
                st.markdown(st.session_state.ai_result_academic)


    except Exception as e:
        st.error(f"ファイルの読み込みまたは分析中にエラーが発生しました: {e}")

