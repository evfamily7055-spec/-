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
from streamlit_extras.row import row # --- 属性別特徴語の表示改善のために追加 ---
import prince # --- 対応分析のために追加 ---

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
@st.cache_data # テキストとTokenizerに変化がなければキャッシュ
def extract_words(text, _tokenizer): # _tokenizer引数はキャッシュのキーとして使う
    if not isinstance(text, str):
        return []
    tokenizer = get_tokenizer()
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
@st.cache_data # AI分析結果もキャッシュ
def call_gemini_api(text_data, has_attribute):
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

# --- 6. 対応分析 関数 ---
@st.cache_data
def run_correspondence_analysis(_df, attribute_col): # _df を受け取る
    try:
        # 最も頻繁に出現する単語 Top 30 を選定
        all_words_list = [word for sublist in _df['words'] for word in sublist]
        top_words = [word for word, freq in Counter(all_words_list).most_common(30)]
        
        # 属性のユニークな値を取得
        unique_attrs = _df[attribute_col].dropna().unique()
        
        if len(unique_attrs) < 2:
            return None, "対応分析エラー: 比較対象の属性が2つ未満です。"
        if not top_words:
            return None, "対応分析エラー: 分析対象の単語が見つかりません。"

        # 「単語 × 属性」のクロス集計表を作成
        cross_tab = pd.DataFrame(index=top_words, columns=unique_attrs)
        for word in top_words:
            for attr in unique_attrs:
                # その属性を持ち、かつその単語を含む文書の数
                count = sum(1 for i, row in _df[_df[attribute_col] == attr].iterrows() if word in row['words'])
                cross_tab.loc[word, attr] = count
        
        # NaNを0で埋める
        cross_tab = cross_tab.fillna(0).astype(int)
        
        # 合計が0の行・列があるとエラーになるため削除
        cross_tab = cross_tab.loc[cross_tab.sum(axis=1) > 0]
        cross_tab = cross_tab.loc[:, cross_tab.sum(axis=0) > 0]
        
        if cross_tab.empty:
            return None, "対応分析エラー: 有効なクロス集計表を作成できませんでした。"

        # 対応分析を実行
        ca = prince.CA(
            n_components=2,
            n_iter=3,
            copy=True,
            check_input=True,
            random_state=42
        )
        ca = ca.fit(cross_tab)

        # プロットを作成
        fig = ca.plot(
            cross_tab,
            x_component=0,
            y_component=1,
            show_row_labels=True,
            show_col_labels=True,
            figsize=(12, 12)
        )
        
        return fig, None

    except Exception as e:
        return None, f"対応分析エラー: {e}"

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
            return fig_wc, None
        else:
            wc = WordCloud(width=800, height=400, background_color='white',
                           font_path=font_path).generate_from_frequencies(word_freq)
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
        unique_words = sorted(list(set(words)))
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


# --- 9. メイン画面のUI ---
uploaded_file = st.file_uploader("1. Excelファイル (xlsx) をアップロード", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        # --- session_stateの初期化 ---
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
                        st.session_state.pop('ai_result', None)
                        st.session_state.pop('fig_wc', None)
                        st.session_state.pop('fig_net', None)
                        st.session_state.pop('chi2_results', None)
                        st.session_state.pop('fig_ca', None)

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

            tab_names = ["🤖 AI サマリー", "☁️ WordCloud", "🕸️ 共起ネットワーク", "🔍 KWIC (文脈検索)", "📈 属性別 特徴語", "🗺️ 対応分析"]
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
            
            # --- Tab 1: AI サマリー ---
            with tab1:
                if 'ai_result' not in st.session_state:
                    with st.spinner("AIによる要約を生成中..."):
                        sample_df = df_analyzed.sample(n=min(len(df_analyzed), 100))
                        
                        def format_for_ai(row):
                            text = row[text_column] or ''
                            attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                            if attrs:
                                return f"[{' | '.join(attrs)}] || {text}"
                            return text
                            
                        ai_input_text = "\n".join(sample_df.apply(format_for_ai, axis=1))
                        st.session_state.ai_result = call_gemini_api(ai_input_text, has_attribute=bool(attribute_columns))
                
                st.markdown(st.session_state.ai_result)
            
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
                else:
                    st.warning(st.session_state.wc_error)
                
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
                                    wc = WordCloud(width=600, height=300, background_color='white',
                                                   font_path=font_path).generate_from_frequencies(Counter(subset_words_list))
                                    ax.imshow(wc, interpolation='bilinear')
                                    ax.axis('off')
                                except Exception as e:
                                    st.error(f"WordCloud(属性:{val})の生成に失敗: {e}")
                            
                            st.pyplot(fig_subset_wc)
            
            # --- Tab 3: 共起ネットワーク ---
            with tab3:
                if 'fig_net' not in st.session_state:
                     with st.spinner("共起ネットワークを計算中..."):
                        fig_net, net_error = generate_network(df_analyzed['words'], font_path)
                        st.session_state.fig_net = fig_net
                        st.session_state.net_error = net_error

                if st.session_state.fig_net:
                    st.pyplot(st.session_state.fig_net)
                else:
                    st.warning(st.session_state.net_error)
            
            # --- Tab 4: KWIC (文脈検索) ---
            with tab4: 
                st.subheader("KWIC (文脈検索)")
                st.info("キーワードに `*` を含めるとワイルドカード検索が可能です (例: `顧客*`)。")
                kwic_keyword = st.text_input("文脈を検索したい単語を入力してください", key="kwic_input")
                if kwic_keyword:
                    # KWICは毎回再計算（キャッシュしない）
                    kwic_html = generate_kwic_html(df_analyzed, text_column, kwic_keyword)
                    st.markdown(kwic_html, unsafe_allow_html=True) 
            
            # --- Tab 5: 属性別 特徴語 ---
            with tab5: 
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
                        if not chi2_results:
                            st.info(f"属性「{attr_col_for_chi2}」には、統計的に有意な特徴語は見つかりませんでした。")
                        else:
                            r = row(*[1 for _ in range(len(chi2_results))], vertical_align="top")
                            for i, (attr_value, words) in enumerate(chi2_results.items()):
                                with r.columns[i]:
                                    st.markdown(f"**{attr_value}** の特徴語 (Top 20)")
                                    if words:
                                        for word, p_value, chi2_val in words:
                                            st.write(f"- {word} (p={p_value:.3f})")
                                    else:
                                        st.info("特徴語なし")
            
            # --- Tab 6: 対応分析 ---
            with tab6: 
                st.subheader("対応分析（コレスポンデンス分析）")
                if not attribute_columns:
                    st.warning("この分析を行うには、サイドバーで「分析軸」を1つ以上選択してください。")
                else:
                    attr_col_for_ca = attribute_columns[0]
                    st.info(f"属性 「**{attr_col_for_ca}**」 と、頻出単語 (Top 30) との関係性をプロットします。")
                    
                    if 'fig_ca' not in st.session_state:
                        with st.spinner("対応分析を計算中..."):
                            fig_ca, ca_error = run_correspondence_analysis(df_analyzed, attr_col_for_ca)
                            st.session_state.fig_ca = fig_ca
                            st.session_state.ca_error = ca_error

                    if st.session_state.ca_error:
                        st.error(st.session_state.ca_error)
                    elif st.session_state.fig_ca:
                        st.pyplot(st.session_state.fig_ca)
                    else:
                        st.warning("対応分析のプロットを生成できませんでした。")

    except Exception as e:
        st.error(f"ファイルの読み込みまたは分析中にエラーが発生しました: {e}")

