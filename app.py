import streamlit as st
import pandas as pd
import re
import requests # Gemini API呼び出し用
import time # リトライ用
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import japanize_matplotlib # Matplotlibの日本語化

# --- 1. アプリの基本設定 ---
st.set_page_config(page_title="統計＋AI 統合アナライザー", layout="wide")
st.title("統計＋AI 統合テキストアナライザー 📊🤖")
st.write("Excelをアップロードし、テキスト列と分析軸（属性）を選択してください。統計分析とAIによる要約を同時に実行します。")

# --- 2. 形態素解析＆ストップワード設定 (キャッシュ) ---
@st.cache_resource # 形態素解析器は重いのでキャッシュ
def get_tokenizer():
    return Tokenizer()

# ストップワード（一般的すぎる単語）
STOPWORDS = set([
    'の', 'に', 'は', 'を', 'た', 'です', 'ます', 'が', 'で', 'も', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'する',
    'ない', 'こと', 'もの', 'これ', 'それ', 'あれ', 'よう', 'ため', '人', '中', '等', '思う', 'いう', 'なる', '日', '時'
])

# テキストを単語リストに分割する関数
def extract_words(text, tokenizer):
    if not isinstance(text, str):
        return []
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        part_of_speech = token.part_of_speech.split(',')[0]
        # 名詞、動詞、形容詞のみを抽出
        if part_of_speech in ['名詞', '動詞', '形容詞']:
            # ストップワードでなく、1文字以上なら追加
            if token.base_form not in STOPWORDS and len(token.base_form) > 1:
                words.append(token.base_form)
    return words

# --- 3. Gemini AI 分析関数 ---
def call_gemini_api(text_data, has_attribute):
    apiKey = "" # APIキーは空のままにしてください
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"

    attributeInstruction = has_attribute \
        ? "データは「属性 || テキスト」の形式です。属性ごとの傾向や違いにも着目して分析してください。" \
        : ""
    
    systemPrompt = f"""あなたは、テキストマイニングの専門家です。与えられたテキスト群を分析し、結果を詳細かつ分かりやすくマークダウン形式で出力してください。
{attributeInstruction}
## 1. 分析サマリー
(全体の傾向を簡潔に要約)
## 2. 主要なテーマ
(頻出するトピックや意見のカテゴリを3〜5個提示)
## 3. ポジティブな意見
(具体的な良い点を引用しつつリストアップ)
## 4. ネガティブな意見・課題
(具体的な不満や改善点を引用しつつリストアップ)
{has_attribute and '## 5. 属性別の傾向 (もしあれば)\n(属性ごとの特徴的な意見を比較)'}
## 6. 総評とネクストアクション
(分析から言えること、次に行うべきアクションを提案)
"""
    
    payload = {
        "systemInstruction": {"parts": [{"text": systemPrompt}]},
        "contents": [{"parts": [{"text": textData}]}]
    }

    try:
        response = None
        delay = 1000
        for i in range(5):
            response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                break
            elif response.status_code == 429 or response.status_code >= 500:
                time.sleep(delay / 1000)
                delay *= 2
            else:
                response.raise_for_status() # 他のエラー
        
        if response.status_code != 200:
             return f"AI分析に失敗しました。リトライ後もエラーが解消しませんでした。 (Status: {response.status_code})"

        result = response.json()
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        return text if text else "AIからの応答が空でした。"

    except Exception as e:
        return f"AI分析中にエラーが発生しました: {e}"

# --- 4. メイン画面のUI ---
uploaded_file = st.file_uploader("1. Excelファイル (xlsx) をアップロード", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("読み込みデータ (先頭5件)")
        st.dataframe(df.head())
        
        all_columns = df.columns.tolist()
        
        # --- 5. 分析設定 (サイドバー) ---
        with st.sidebar:
            st.header("⚙️ 分析設定")
            text_column = st.selectbox("分析したいテキスト列", all_columns, index=0)
            
            # 複数属性の選択に対応！
            attribute_columns = st.multiselect("分析軸 (複数OK: 例: 年代, 性別)", all_columns)
            
            st.markdown("---")
            run_button = st.button("分析を実行", type="primary", use_container_width=True)

        # --- 6. 分析実行 ---
        if run_button:
            if not text_column:
                st.error("「分析したいテキスト列」を選択してください。")
            else:
                # --- 6a. 統計分析 (形態素解析) ---
                with st.spinner("ステップ1/3: 形態素解析を実行中... (全件処理)"):
                    tokenizer = get_tokenizer()
                    # 'words' 列に、分割した単語リストを追加
                    df['words'] = df[text_column].apply(lambda x: extract_words(x, tokenizer))
                    st.success("形態素解析が完了しました。")

                # --- 6b. AI分析 (サンプリング) ---
                with st.spinner("ステップ2/3: AIによる要約を生成中..."):
                    # AIには全件送ると多すぎるため、100件をサンプリング
                    sample_df = df.sample(n=min(len(df), 100))
                    
                    def format_for_ai(row):
                        text = row[text_column] or ''
                        attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                        if attrs:
                            return f"[{' | '.join(attrs)}] || {text}"
                        return text
                        
                    ai_input_text = "\n".join(sample_df.apply(format_for_ai, axis=1))
                    ai_result = call_gemini_api(ai_input_text, has_attribute=bool(attribute_columns))
                    st.success("AI要約が完了しました。")

                # --- 6c. 統計分析 (WordCloud, Network) ---
                with st.spinner("ステップ3/3: 統計グラフを生成中..."):
                    # 全単語をフラットなリストに
                    all_words_list = [word for sublist in df['words'] for word in sublist]
                    
                    # 1. WordCloud
                    fig_wc = None
                    if all_words_list:
                        word_freq = Counter(all_words_list)
                        try:
                            # Streamlit Cloud には日本語フォントがないため、japanize_matplotlib に期待
                            wc = WordCloud(width=800, height=400, background_color='white',
                                           font_path=None).generate_from_frequencies(word_freq)
                            fig_wc, ax = plt.subplots(figsize=(12, 6))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                        except Exception as e:
                            st.error(f"WordCloudの生成に失敗しました。フォントの問題かもしれません。エラー: {e}")

                    # 2. 共起ネットワーク
                    fig_net = None
                    co_occur_counter = Counter()
                    for words in df['words']:
                        unique_words = sorted(list(set(words)))
                        for w1, w2 in combinations(unique_words, 2):
                            co_occur_counter[(w1, w2)] += 1
                    
                    top_pairs = co_occur_counter.most_common(50)
                    
                    if top_pairs:
                        G = nx.Graph()
                        for (w1, w2), weight in top_pairs:
                            G.add_edge(w1, w2, weight=weight)
                        
                        fig_net, ax = plt.subplots(figsize=(14, 14))
                        pos = nx.spring_layout(G, k=0.8)
                        edge_weights = [d['weight'] * 0.5 for u,v,d in G.edges(data=True)]
                        
                        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', alpha=0.8)
                        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='gray')
                        nx.draw_networkx_labels(G, pos, font_size=10) # japanize_matplotlibが日本語を表示
                        ax.axis('off')
                    
                    st.success("統計グラフの生成が完了しました。")

                # --- 7. 結果表示 (タブ) ---
                st.subheader("📊 分析結果")
                tab1, tab2, tab3 = st.tabs(["🤖 AI サマリー", "☁️ WordCloud", "🕸️ 共起ネットワーク"])
                
                with tab1:
                    st.markdown(ai_result)
                
                with tab2:
                    if fig_wc:
                        st.pyplot(fig_wc)
                    else:
                        st.warning("WordCloudを生成できませんでした。")
                
                with tab3:
                    if fig_net:
                        st.pyplot(fig_net)
                    else:
                        st.warning("共起ネットワークを生成できませんでした。")

    except Exception as e:
        st.error(f"ファイルの読み込みまたは分析中にエラーが発生しました: {e}")

