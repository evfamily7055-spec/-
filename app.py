import streamlit as st
import pandas as pd
import re
import requests # Gemini API呼び出し用
import time # リトライ用
import json # --- JSONパースのために追加 ---
import plotly.graph_objects as go # --- ▼ Plotly をインポート ---
import plotly.express as px # --- ▼ Plotly Express (カラーパレット用) をインポート ---
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
import networkx as nx 
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import Counter
from itertools import combinations
import japanize_matplotlib # Matplotlibの日本語化 (WordCloud と NetworkX で依然として必要)
import numpy as np 
from scipy.stats import chi2_contingency 
import io 
import base64 
from streamlit.components.v1 import html 

# --- 1. アプリの基本設定 ---
st.set_page_config(page_title="統計＋AI 統合アナライザー (Plotly Ver.)", layout="wide")
st.title("統計＋AI 統合テキストアナライザー 📊🤖 (Plotly Ver.)")
st.write("Excelをアップロードし、テキスト列と分析軸（属性）を選択してください。統計分析とAIによる要約・クラスター分析を同時に実行します。")

# --- 2. 形態素解析＆ストップワード設定 (キャッシュ) ---
@st.cache_resource 
def get_tokenizer():
    return Tokenizer()

BASE_STOPWORDS = set([
    'の', 'に', 'は', 'を', 'た', 'です', 'ます', 'が', 'で', 'も', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'する',
    'ない', 'こと', 'もの', 'これ', 'それ', 'あれ', 'よう', 'ため', '人', '中', '等', '思う', 'いう', 'なる', '日', '時',
    'くださる', 'いただく', 'しれる', 'くる', 'おる', 'れる', 'られる', 'せる', 'させる', 'できる', 'なる', 'やる', 'いく',
    '行う', '言う', '申し上げる', 'まいる', '見る', 'ここ', 'そこ', 'あそこ', 'こちら', 'そちら', 'あちら', 'この', 'その', 'あの'
])

# AI分析の最大文字数制限を定義
MAX_AI_INPUT_CHARS = 1000000

@st.cache_data 
def extract_words(text, _tokenizer): 
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
            if len(token.base_form) > 1: 
                words.append(token.base_form)
    return words

# --- 3. Gemini AI 分析関数 (会話対応版) ---

# --- ▼ 修正点: AI サマリー (表形式) プロンプト ---
# 1. シンプルな要約プロンプト (固定)
SYSTEM_PROMPT_SIMPLE = """あなたは、テキストマイニングの専門家です。
与えられた[分析対象テキスト]と、AIによる[クラスター分析結果JSON]の両方を参照し、分析サマリーをマークダウン形式で出力してください。

[クラスター分析結果JSON]
{cluster_json_data}

[あなたのタスク]
{analysis_scope_instruction}
{attributeInstruction}

## 1. 分析サマリー
(全体の傾向を簡潔に要約)

## 2. 主要なテーマ
(**重要**: このセクションは、**上記[クラスター分析結果JSON]に厳密に従い**、以下の**Markdownテーブル形式**で記述してください。)

| 主要テーマ（クラスター） | 割合 (%) | 概要 (サブトピックを含む) |
| :--- | :---: | :--- |
| [クラスターAの名前] | [クラスターAの割合※] | [クラスターAのサブトピックを簡潔にまとめた概要] |
| [クラスターBの名前] | [クラスターBの割合※] | [クラスターBのサブトピックを簡潔にまとめた概要] |
| ... | ... | ... |
※割合は、サブトピックの `value` の合計を % にしたものです。 (例: 15.2)

## 3. ポジティブな意見
(与えられた[分析対象テキスト]全体から、具体的な良い点を引用しつつリストアップしてください。**引用する際は、[行番号: XX] も含めてください。**)

## 4. ネガティブな意見・課題
(与えられた[分析対象テキスト]全体から、具体的な不満や改善点を引用しつつリストアップしてください。**引用する際は、[行番号: XX] も含めてください。**)

## 5. 少数だが注目すべき意見
(与えられた[分析対象テキスト]全体から、件数は少ない（例: 1〜2件）かもしれないが、非常に重要、ユニーク、または示唆に富む意見があれば、ここに抽出してください。**引用する際は、[行番号: XX] も含めてください。**)

{has_attribute}

## 7. 総評とネクストアクション
(分析から言えること、次に行うべきアクションを提案)
"""
# --- ▲ 修正完了 ▲ ---


# 2. 学術論文用プロンプト (固定)
SYSTEM_PROMPT_ACADEMIC = """あなたは、テキストデータを分析する計量テキスト分析（テキストマイニング）の専門家です。
与えられた[分析対象テキスト]と、AIによる[クラスター分析結果JSON]の両方を参照し、学術論文形式で出力してください。

[クラスター分析結果JSON]
{cluster_json_data}

[あなたのタスク]
{analysis_scope_instruction}
{attributeInstruction}
以下の構成に従って、マークダウン形式で記述してください。

## 1. 分析の概要 (Abstract)
(データ全体の主要な傾向や特筆すべき点を、客観的な要約として2〜3文で記述する)

## 2. 主要な知見 (Key Findings)
(**重要**: このセクションは、**上記[クラスター分析結果JSON]に厳密に従って**記述してください。JSONの `children` (主要クラスター) を主要な知見として取り上げ、その内容（サブトピック）と割合について客観的に記述する。**引用する際は、[行番号: XX] も含めること。**)

## 3. その他の注目すべき所見 (Other Notable Findings)
(頻度は低い（例: 1〜2件）ものの、分析上見過ごすべきではない特異な意見、または将来の課題を示唆するような意見があれば、ここに記述する。**引用する際は、[行番号: XX] も含めること。**)
{has_attribute}
## 5. 考察と今後の課題 (Discussion and Limitations)
(分析結果から導かれる考察や示唆を記述する。また、データから見られる潜在的な課題や、さらなる分析の方向性についても言及する)
"""

# --- ▼ 修正点: JSONプロンプト (name に割合を含めない) ---
# 3. クラスター分析 (JSON生成用) のシステムプロンプト
SYSTEM_PROMPT_CLUSTER_JSON = """あなたは高度なテキストクラスタリング専門のアナリストです。提供されたテキストデータを分析し、主要な言説クラスター（3〜5個）と、それらを構成するサブトピック（各クラスター内で3〜5個）に分類してください。
{analysis_scope_instruction}

[タスク]
1. テキスト全体を読み、主要なテーマ（クラスター）を3〜5個特定します。
2. 各クラスターを構成する、より詳細なサブトピックを3〜5個特定します。
3. 各サブトピックが、分析対象データ**全体**（{analyzed_items}件）の中で占めるおおよその割合（%）を計算します。
4. `name` フィールドには `[トピック名]` のみを入れてください（**割合(%)の文字列は含めないでください**）。
5. `value` フィールドには、サブトピックの割合（パーセンテージの数値のみ）を設定してください。
6. **重要**: 出力は、指定されたJSONスキーマに厳密に従ってください。

[例]
{{
  "name": "全体",
  "children": [
    {{
      "name": "クラスターA",
      "children": [
        {{ "name": "サブトピックA1", "value": 15.0 }},
        {{ "name": "サブトピックA2", "value": 10.0 }}
      ]
    }},
    {{
      "name": "クラスターB",
      "children": [
        {{ "name": "サブトピックB1", "value": 20.0 }}
      ]
    }}
  ]
}}
"""
# --- ▲ 修正完了 ▲ ---

# 4. クラスター分析 (テキスト解釈用) のシステムプロンプト
SYSTEM_PROMPT_CLUSTER_TEXT = """あなたはテキストアナリストです。以下のJSONは、テキストデータをクラスター分析した結果です。
{analysis_scope_instruction}

[分析結果JSON]
{json_data}

[あなたのタスク]
このJSONデータを解釈し、分析結果をマークダウン形式で分かりやすく説明してください。
**必ず以下の構成に従ってください。**

## 凡例 (色と主要クラスター)
（グラフの各色（例: 薄い青、薄い緑など）が、どの主要クラスターに対応しているかをリスト形式で説明してください。**色は自動で割り当てられています。**）
- [色1 (例: 薄い青)]: [クラスターAの名前]
- [色2 (例: 薄い緑)]: [クラスターBの名前]
- [色3 (例: 薄いオレンジ)]: [クラスターCの名前]
...

## AIによるクラスターの解釈
（次に、各主要クラスターがどのような意見グループなのかを詳細に説明してください。サブトピックにも触れながら、なぜそのように分類されたのかを具体的に考察してください。）
"""


# 5. 会話用プロンプト (可変)
SYSTEM_PROMPT_CHAT = """あなたは、与えられたテキストデータ（コンテキスト）に関する質問に答える、優秀なデータアナリストです。
コンテキストは `[行番号: XX] [属性...] || テキスト` の形式で提供されます。
ユーザーからの質問に対し、提供されたコンテキスト情報に基づいて、簡潔かつ的確に回答してください。
コンテキストに含まれていない情報については、その旨を正直に伝えてください。
"""

# 6. 感情分析 (JSON生成用) のシステムプロンプト
SYSTEM_PROMPT_SENTIMENT_JSON = """あなたは高度な感情分析専門のアナリストです。
{analysis_scope_instruction}

[タスク]
1. 提供された[分析対象テキスト]の全件を読み込みます。
2. 各テキストが「ポジティブ」「ネガティブ」「中立」のどれに該当するかを分類します。
3. 最終的に、3つのカテゴリそれぞれの**合計件数 (count)** と、分析対象データ全体（{analyzed_items}件）に占める**割合 (percentage)** を計算します。
4. **重要**: `count` の合計は、分析対象の総件数 {analyzed_items} 件と一致する必要があります。
5. **重要**: 出力は、指定されたJSONスキーマに厳密に従ってください。
"""


def call_gemini_api(contents, system_instruction=None, generation_config=None):
    try: apiKey = st.secrets["GEMINI_API_KEY"]
    except Exception: return "AI分析エラー: Streamlit CloudのSecretsに `GEMINI_API_KEY` が設定されていません。"
    if not apiKey: return "AI分析エラー: Streamlit CloudのSecretsに `GEMINI_API_KEY` が設定されていません。"

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"

    payload = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    
    if generation_config:
        payload["generationConfig"] = generation_config

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

# --- Plotly Treemap 用のデータ変換関数 ---
def parse_json_for_plotly(json_data_str):
    try:
        data = json.loads(json_data_str)
    except Exception as e:
        return None, None, None, f"JSON解析エラー: {e}"

    labels = []
    parents = []
    values = []
    
    root_name = data.get('name', '全体')
    labels.append(root_name)
    parents.append("")
    values.append(0) 
    
    clusters = data.get('children', [])
    if not clusters:
        return None, None, None, "JSONに 'children' (クラスター) が見つかりません。"

    for cluster in clusters:
        # AIは割合を含まない名前 (例: "クラスターA") を返す
        cluster_name = cluster.get('name', '不明なクラスター') 
        labels.append(cluster_name)
        parents.append(root_name)
        
        sub_topics = cluster.get('children', [])
        
        if not sub_topics:
             values.append(1) 
        else:
            cluster_total_value = 0
            for sub_topic in sub_topics:
                # AIは割合を含まない名前 (例: "サブトピックA1") を返す
                sub_name = sub_topic.get('name', '不明なトピック')
                sub_value = sub_topic.get('value', 0)
                
                if sub_value > 0:
                    labels.append(sub_name)
                    parents.append(cluster_name)
                    values.append(sub_value)
                    cluster_total_value += sub_value
            
            # 親クラスター (深さ1) の value を設定 (0 = 自動集計)
            values.append(0) 


    return labels, parents, values, None


# --- ▼ 修正点: `Plotly` を使ったTreemap描画関数 (ビジュアル改善) ---
def create_plotly_treemap(json_data_str):
    """
    AIが生成したJSONデータから、Plotlyを使用して
    インタラクティブなTreemapを生成します。(ビジュアル改善版)
    """
    labels, parents, values, error = parse_json_for_plotly(json_data_str)
    
    if error:
        return None, error
        
    root_label = labels[0] # "全体"
    clusters = [l for l, p in zip(labels, parents) if p == root_label]
    num_clusters = len(clusters)

    # 柔らかいパステルカラーのパレット "Set3" を使用
    if num_clusters > 0:
        colors = px.colors.qualitative.Set3[:num_clusters]
    else:
        colors = px.colors.qualitative.Set3

    fig = go.Figure(go.Treemap(
        labels = labels,
        parents = parents,
        values = values,
        
        # ▼ 修正点: テキストテンプレートでラベル(太字)と割合(改行)を指定
        # %{percentRoot} は全体 (root) に対する割合
        texttemplate="<b>%{label}</b><br>%{percentRoot:.1%}",
        
        hoverinfo="label+value+percent root", 
        
        # ▼ 修正点: 親(クラスター)ごとに色を割り当て (Set3 パレット)
        marker_colorscale='Set3',
        branchvalues="total", # 親の合計が子の合計になるように
        
        # ▼ 修正点: テキストの自動調整と重なり防止
        # 領域に収まるようにフォントサイズを自動調整
        # 最小サイズを10ptに設定し、それより小さくなる場合はテキストを非表示にする
        uniformtext=dict(minsize=10, mode='hide'), 
        
        pathbar_textfont={'size': 16}
    ))
    
    fig.update_layout(
        margin = dict(t=50, l=10, r=10, b=10),
        title_text="トピック構成 (Treemap)",
        title_font_size=20,
        # ▼ 修正点: カラーウェイを "Set3" に明示的に指定
        colorway=px.colors.qualitative.Set3 
    )
    
    return fig, None
# --- ▲ 修正完了 ▲ ---


# --- 感情分析円グラフ描画関数 ---
def create_sentiment_pie_chart(json_data_str):
    try:
        data = json.loads(json_data_str)
        if not isinstance(data, list) or len(data) == 0:
             return None, "AIが生成したJSONの形式が不正です (リストではありません)。"
    except json.JSONDecodeError:
        return None, "AIが生成したJSONの解析に失敗しました。"
    except TypeError:
         return None, "AIの応答が空または不正です。"

    labels = []
    sizes = []
    colors = []
    
    color_map = {
        "ポジティブ": "#4CAF50", # 緑
        "ネガティブ": "#F44336", # 赤
        "中立": "#9E9E9E"       # グレー
    }
    
    for item in data:
        sentiment = item.get("sentiment", "不明")
        count = item.get("count", 0)
        
        if count > 0: 
            labels.append(f"{sentiment}\n({count:,}件)")
            sizes.append(count)
            colors.append(color_map.get(sentiment, "#BDBDBD")) 

    if not sizes:
        return None, "描画対象となる感情データ（件数 > 0）が見つかりませんでした。"

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct=lambda p: f"{p:.1f}%\n({int(p/100.*sum(sizes)):,d}件)",
            startangle=90,
            pctdistance=0.85, 
            labeldistance=1.1, 
            textprops={'color':'black', 'fontsize': 11} 
        )
        
        plt.setp(autotexts, color='white', fontweight='bold', fontsize=10)
        
        ax.set_title("感情分析（ポジ・ネガ・中立）の割合", fontsize=18)
        ax.axis('equal')  
        
        plt.close(fig) 
        return fig, None
    except Exception as e:
        return None, f"円グラフ描画中にエラーが発生: {e}"


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

# --- 8. 共起ネットワーク生成関数 (改善版) ---
def generate_network(_words_df, font_path, _stopwords_set):
    
    co_occur_counter = Counter()
    for words in _words_df:
        unique_words = sorted(list(set(word for word in words if word not in _stopwords_set)))
        for w1, w2 in combinations(unique_words, 2): co_occur_counter[(w1, w2)] += 1
    
    top_pairs = co_occur_counter.most_common(70) 
    
    if not top_pairs:
        return None, "共起ネットワーク生成不可（共起ペア不足）。"

    G = nx.Graph()
    for (w1, w2), weight in top_pairs:
        G.add_edge(w1, w2, weight=weight)
        
    all_words_in_docs = [word for sublist in _words_df for word in sublist if word not in _stopwords_set]
    all_word_freq = Counter(all_words_in_docs)
    
    nodes_in_graph = list(G.nodes())
    node_sizes = []
    for node in nodes_in_graph:
        size = all_word_freq.get(node, 1) * 30 
        node_sizes.append(max(500, min(size, 5000))) 

    try:
        communities_generator = greedy_modularity_communities(G)
        communities = sorted(communities_generator, key=len, reverse=True)
        
        community_map = {}
        # ネットワークの色もパステル (Set3) に変更
        cmap = plt.get_cmap('Set3', len(communities)) 
        
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = cmap(i)
        
        node_colors = [community_map[node] for node in G.nodes()]
    except Exception:
        node_colors = 'lightblue' 

    edge_weights = [d['weight'] * 0.3 for u,v,d in G.edges(data=True)] 

    try:
        fig_net, ax = plt.subplots(figsize=(18, 18)); 
        
        pos = nx.spring_layout(G, k=1.0, iterations=50) 
        
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_size=node_sizes,    
            node_color=node_colors   
        )
        
        nx.draw_networkx_edges(
            G, 
            pos, 
            width=edge_weights,     
            alpha=0.4, 
            edge_color='gray'
        )
        
        labels_kwargs = {'font_size': 9, 'font_family': 'IPAexGothic'} if font_path else {'font_size': 9}
        nx.draw_networkx_labels(G, pos, **labels_kwargs)
        
        ax.axis('off')
        plt.close(fig_net) 
        return fig_net, None
        
    except Exception as e:
        return None, f"ネットワーク描画エラー: {e}"


# 単語頻度計算関数
def calculate_frequency(_words_list, _stopwords_set, top_n=50):
    filtered_words = [word for word in _words_list if word not in _stopwords_set]
    if not filtered_words: return pd.DataFrame(columns=['Rank', 'Word', 'Frequency'])
    word_freq = Counter(filtered_words).most_common(top_n)
    freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    freq_df['Rank'] = freq_df.index + 1
    return freq_df[['Rank', 'Word', 'Frequency']]

# HTMLレポート生成関数
def fig_to_bytes(fig):
    if fig is None: return None
    if isinstance(fig, plt.Figure):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()
    elif isinstance(fig, go.Figure):
        try:
            return fig.to_image(format="png", width=1200, height=700, scale=2)
        except ImportError:
            st.error("HTMLレポートへの画像埋め込みには `kaleido` が必要です。`requirements.txt` に `kaleido` を追加してください。")
            return None
        except Exception as e:
            st.error(f"Plotly画像の書き出しエラー: {e}")
            return None
    return None

def fig_to_base64_png(fig):
    img_bytes = fig_to_bytes(fig)
    if img_bytes is None: return None
    return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

def generate_html_report():
    html_parts = ["<!DOCTYPE html><html lang='ja'><head><meta charset='UTF-8'><title>テキスト分析レポート</title>"]
    html_parts.append("<style>body{font-family:sans-serif;margin:20px}h1,h2,h3{color:#333;border-bottom:1px solid #ccc;padding-bottom:5px}h2{margin-top:30px}.result-section{margin-bottom:30px;padding:15px;border:1px solid #eee;border-radius:5px;background-color:#f9f9f9}img{max-width:100%;height:auto;border:1px solid #ddd;margin-top:10px}table{border-collapse:collapse;width:100%;margin-top:10px}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}pre{background-color:#eee;padding:10px;border-radius:3px;white-space:pre-wrap;word-wrap:break-word}</style>")
    html_parts.append("</head><body><h1>テキスト分析レポート</h1>")
    if 'ai_result_simple' in st.session_state: html_parts.append(f"<div class='result-section'><h2>🤖 AI サマリー (簡易)</h2><pre>{st.session_state.ai_result_simple}</pre></div>")
    
    if 'fig_sentiment_pie_display' in st.session_state and st.session_state.fig_sentiment_pie_display:
        img_base64 = fig_to_base64_png(st.session_state.fig_sentiment_pie_display);
        if img_base64: html_parts.append(f"<div class='result-section'><h2>💖 AI 感情分析</h2><img src='{img_base64}' alt='Sentiment Pie Chart'></div>")

    if 'fig_treemap_display' in st.session_state and st.session_state.fig_treemap_display:
        img_base64 = fig_to_base64_png(st.session_state.fig_treemap_display);
        if img_base64: html_parts.append(f"<div class='result-section'><h2>📊 AI クラスター分析 (Treemap)</h2><img src='{img_base64}' alt='Treemap'></div>")
    if 'ai_result_cluster_text' in st.session_state: html_parts.append(f"<div class='result-section'><h2>📊 AI クラスター分析 (解釈)</h2><pre>{st.session_state.ai_result_cluster_text}</pre></div>")

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

# def fig_to_bytes(fig): ... (上へ移動)

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
            
            st.info(f"AI分析は全件（最大{MAX_AI_INPUT_CHARS:,}文字）を対象とします。統計分析は常に全件が対象です。", icon="ℹ️")

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
                        
                        st.session_state.pop('ai_result_simple', None); st.session_state.pop('ai_result_academic', None)
                        st.session_state.pop('ai_result_cluster_json', None)
                        st.session_state.pop('ai_result_cluster_text', None)
                        st.session_state.pop('fig_treemap_display', None) 
                        st.session_state.pop('treemap_error_display', None)
                        st.session_state.pop('ai_result_sentiment_json', None) 
                        st.session_state.pop('fig_sentiment_pie_display', None)
                        st.session_state.pop('sentiment_pie_error_display', None) 
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

            # --- タブを10個に増やす ---
            tab_names = ["🤖 AI サマリー", "💖 AI 感情分析", "📊 AI クラスター", "☁️ WordCloud", "📊 単語頻度", "🕸️ 共起ネットワーク", "🔍 KWIC", "📈 属性別特徴語", "📝 AI 学術論文", "💬 AI チャット"]
            tab1, tab_sentiment, tab_cluster, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names)
            
            # --- (共通) AIに渡すテキストと件数を生成するロジック ---
            
            def format_for_ai(row):
                excel_row_num = row.name + 2 
                text = row[text_column] or ''; 
                attrs = [str(row[col] or 'N/A') for col in attribute_columns]
                id_str = f"[行番号: {excel_row_num}]"
                attr_str = f"[{' | '.join(attrs)}]" if attrs else ""
                return f"{id_str} {attr_str} || {text}"

            total_items = len(df_analyzed)
            ai_input_parts = []
            current_char_count = 0
            analyzed_items = 0

            for index, row in df_analyzed.iterrows():
                row_text = format_for_ai(row) + "\n" 
                if current_char_count + len(row_text) > MAX_AI_INPUT_CHARS:
                    break
                ai_input_parts.append(row_text)
                current_char_count += len(row_text)
                analyzed_items += 1
            
            ai_input_text = "".join(ai_input_parts)
            
            if analyzed_items < total_items:
                analysis_scope_instr = f"【重要】全 {total_items:,} 件中、先頭の {analyzed_items:,} 件のデータが提供されています。分析や件数・割合の計算は、この {analyzed_items:,} 件のデータを「全体」として行ってください。"
                analysis_scope_warning = f"データが非常に大きいため、AI分析は先頭の {analyzed_items:,} 件（全 {total_items:,} 件中）を対象に実行されました。全件の厳密な統計は他のタブをご覧ください。"
            else:
                analysis_scope_instr = f"【重要】全 {total_items:,} 件のデータが提供されています。分析や件数・割合の計算は、この {total_items:,} 件のデータを「全体」として行ってください。"
                analysis_scope_warning = f"AI分析は全 {total_items:,} 件を対象に実行されました。"
            # --- (共通ロジックここまで) ---


            # --- (共通) クラスターJSONを（必要なら生成しつつ）取得するヘルパー関数 ---
            def get_or_generate_cluster_json(ai_input_text, analysis_scope_instr, analyzed_items):
                if 'ai_result_cluster_json' not in st.session_state:
                    contents_json = [{"parts": [{"text": ai_input_text}]}]
                    schema = {
                        "type": "OBJECT",
                        "properties": {
                            "name": {"type": "STRING", "description": "常に '全体' または 'All Topics'"},
                            "children": {
                                "type": "ARRAY",
                                "description": "主要なクラスター（3〜5個）の配列",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "name": {"type": "STRING", "description": "クラスター名 (例: 'ポジティブな意見')"},
                                        "children": {
                                            "type": "ARRAY",
                                            "description": "サブトピック（3〜5個）の配列",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "name": {"type": "STRING", "description": "サブトピック名 (例: 'デザインへの言及')"},
                                                    "value": {"type": "NUMBER", "description": "サブトピックの割合（数値のみ）"}
                                                },
                                                "required": ["name", "value"]
                                            }
                                        }
                                    },
                                    "required": ["name", "children"]
                                }
                            }
                        },
                        "required": ["name", "children"]
                    }
                    
                    gen_config_json = {
                        "response_mime_type": "application/json",
                        "response_schema": schema
                    }
                    
                    system_instr_json = SYSTEM_PROMPT_CLUSTER_JSON.format(
                        analysis_scope_instruction=analysis_scope_instr,
                        analyzed_items=analyzed_items
                    )
                    
                    json_str = call_gemini_api(contents_json, system_instruction=system_instr_json, generation_config=gen_config_json)
                    st.session_state.ai_result_cluster_json = json_str
                
                return st.session_state.ai_result_cluster_json


            # --- Tab 1: AI サマリー (簡易) ---
            with tab1:
                if 'ai_result_simple' not in st.session_state:
                    with st.spinner("AIによるクラスター分析と要約を生成中..."):
                        
                        if analyzed_items < total_items: st.warning(analysis_scope_warning, icon="⚠️")
                        else: st.info(analysis_scope_warning, icon="✅")

                        try:
                            cluster_json_str = get_or_generate_cluster_json(ai_input_text, analysis_scope_instr, analyzed_items)
                        except Exception as e:
                            st.error(f"クラスターJSONの生成に失敗しました: {e}")
                            cluster_json_str = '{"name": "エラー", "children": []}'

                        contents = [{"parts": [{"text": ai_input_text}]}]
                        has_attr = bool(attribute_columns)
                        has_attribute_str_s = "## 6. 属性別の傾向 (もしあれば)\n(属性ごとの特徴的な意見を比較)" if has_attr else ""
                        attr_instr_s = "データは「属性 || テキスト」の形式です。属性ごとの傾向や違いにも着目して分析してください。" if has_attr else ""
                        
                        system_instr_s = SYSTEM_PROMPT_SIMPLE.format(
                            analysis_scope_instruction=analysis_scope_instr,
                            attributeInstruction=attr_instr_s, 
                            has_attribute=has_attribute_str_s,
                            cluster_json_data=cluster_json_str 
                        )
                        st.session_state.ai_result_simple = call_gemini_api(contents, system_instruction=system_instr_s)
                st.markdown(st.session_state.ai_result_simple)

            # --- (新設) AI 感情分析タブ (JSON + Matplotlib Pie Chart) ---
            with tab_sentiment:
                st.subheader("AIによる感情分析（円グラフ）")
                st.info("AIが全テキストを「ポジティブ」「ネガティブ」「中立」に分類し、その構成比（件数と割合）を可視化します。")
                st.warning("この分析は、AIサマリーの「ポジティブ/ネガティブな意見」の抜粋とは異なり、全件を対象としたAIによる分類集計です。", icon="ℹ️")

                # 1. 感情分析JSONの生成 (キャッシュ確認)
                if 'ai_result_sentiment_json' not in st.session_state:
                    with st.spinner("AIによる感情分析JSONを生成中... (ステップ1/2)"):
                        if analyzed_items < total_items: st.warning(analysis_scope_warning, icon="⚠️")
                        else: st.info(analysis_scope_warning, icon="✅")

                        contents_json = [{"parts": [{"text": ai_input_text}]}]
                        schema = {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "sentiment": {"type": "STRING", "description": "感情ラベル (ポジティブ, ネガティブ, 中立)"},
                                    "count": {"type": "NUMBER", "description": "該当する件数"},
                                    "percentage": {"type": "NUMBER", "description": "全体に占める割合 (xx.x)"}
                                },
                                "required": ["sentiment", "count", "percentage"]
                            }
                        }
                        
                        gen_config_json = {
                            "response_mime_type": "application/json",
                            "response_schema": schema
                        }
                        
                        system_instr_json = SYSTEM_PROMPT_SENTIMENT_JSON.format(
                            analysis_scope_instruction=analysis_scope_instr,
                            analyzed_items=analyzed_items
                        )
                        
                        json_str = call_gemini_api(contents_json, system_instruction=system_instr_json, generation_config=gen_config_json)
                        st.session_state.ai_result_sentiment_json = json_str
                
                # 2. 円グラフの描画 (キャッシュ確認)
                if 'fig_sentiment_pie_display' not in st.session_state and 'ai_result_sentiment_json' in st.session_state:
                    with st.spinner("感情分析円グラフを生成中... (ステップ2/2)"):
                        json_data_str = st.session_state.ai_result_sentiment_json
                        fig_pie, pie_error = create_sentiment_pie_chart(json_data_str)
                        st.session_state.fig_sentiment_pie_display = fig_pie
                        st.session_state.sentiment_pie_error_display = pie_error
                
                # 3. 描画
                if 'fig_sentiment_pie_display' in st.session_state and st.session_state.fig_sentiment_pie_display:
                    fig_pie = st.session_state.fig_sentiment_pie_display
                    st.pyplot(fig_pie)
                    
                    img_bytes = fig_to_bytes(fig_pie)
                    if img_bytes: st.download_button("この画像をダウンロード (PNG)", img_bytes, "sentiment_pie_chart.png", "image/png")

                elif 'sentiment_pie_error_display' in st.session_state:
                    st.error(st.session_state.sentiment_pie_error_display)
                    if 'ai_result_sentiment_json' in st.session_state:
                         st.text_area("AIのJSONレスポンス (エラー)", st.session_state.ai_result_sentiment_json, height=200)
                else:
                    st.info("感情分析データを生成中です...")


            # --- AI クラスター分析タブ (JSON + Plotly Treemap) ---
            with tab_cluster:
                st.subheader("AIによる言説クラスター分析 (Treemap)")
                st.info("AIがテキストを階層的なトピックに分類し、その構成比（面積）を可視化します。グラフ右上のカメラアイコンから画像を保存できます。")

                # 1. JSONデータの生成 (キャッシュ確認)
                if 'ai_result_cluster_json' not in st.session_state:
                    with st.spinner("AIによるクラスターJSONを生成中... (ステップ1/3)"):
                        if analyzed_items < total_items: st.warning(analysis_scope_warning, icon="⚠️")
                        else: st.info(analysis_scope_warning, icon="✅")
                        
                        try:
                            # 共通関数を呼び出す
                            get_or_generate_cluster_json(ai_input_text, analysis_scope_instr, analyzed_items)
                        except Exception as e:
                            st.error(f"クラスターJSONの生成に失敗しました: {e}")
                            if 'ai_result_cluster_json' in st.session_state: 
                                 st.text_area("AIの生レスポンス (エラー)", st.session_state.ai_result_cluster_json, height=100)


                # 2. テキスト解釈の生成 (キャッシュ確認)
                if 'ai_result_cluster_text' not in st.session_state and 'ai_result_cluster_json' in st.session_state:
                     with st.spinner("AIによるクラスターの解釈を生成中... (ステップ2/3)"):
                        json_str = st.session_state.ai_result_cluster_json
                        
                        system_instr_text = SYSTEM_PROMPT_CLUSTER_TEXT.format(
                            analysis_scope_instruction=analysis_scope_instr,
                            json_data=json_str
                        )
                        contents_text = [{"parts": [{"text": "このクラスター分析の結果を、マークダウン形式で詳細に解釈・要約してください。"}]}]
                        
                        text_summary = call_gemini_api(contents_text, system_instruction=system_instr_text)
                        st.session_state.ai_result_cluster_text = text_summary
                
                # 3. Treemap (Plotly) の描画 (キャッシュ確認)
                if 'fig_treemap_display' not in st.session_state and 'ai_result_cluster_json' in st.session_state:
                    with st.spinner("Treemapを生成中... (ステップ3/3)"):
                        json_data_str = st.session_state.ai_result_cluster_json
                        fig_treemap, treemap_error = create_plotly_treemap(json_data_str) # Plotly関数を呼び出す
                        st.session_state.fig_treemap_display = fig_treemap
                        st.session_state.treemap_error_display = treemap_error

                # 4. 描画とテキスト表示
                if 'fig_treemap_display' in st.session_state and st.session_state.fig_treemap_display:
                    st.subheader("トピック構成 (Treemap)")
                    fig_treemap = st.session_state.fig_treemap_display
                    st.plotly_chart(fig_treemap, use_container_width=True) # st.pyplot -> st.plotly_chart
                    
                    if 'ai_result_cluster_text' in st.session_state:
                        # 凡例と解釈はAIの応答に任せる
                        st.markdown(st.session_state.ai_result_cluster_text)
                    else:
                        st.info("クラスターの解釈を生成中です...")
                
                elif 'treemap_error_display' in st.session_state:
                    st.error(st.session_state.treemap_error_display)
                    if 'ai_result_cluster_json' in st.session_state:
                         st.text_area("AIのJSONレスポンス", st.session_state.ai_result_cluster_json, height=200)
                else:
                    st.info("クラスター分析データを生成中です...")
            
            # --- Tab 2: WordCloud --- (tab2 に変更)
            with tab2:
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

            # --- Tab 3: 単語頻度ランキング --- (tab3 に変更)
            with tab3:
                st.subheader("全体の単語頻度ランキング (Top 50)")
                if 'overall_freq_df_display' not in st.session_state:
                     with st.spinner("全体の単語頻度を計算中..."):
                        all_words_list = [word for sublist in df_analyzed['words'] for word in sublist]
                        overall_freq_df = calculate_frequency(all_words_list, current_stopwords_set)
                        st.session_state.overall_freq_df_display = overall_freq_df
                st.dataframe(st.session_state.overall_freq_df_display, use_container_width=True)

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

            # --- Tab 4: 共起ネットワーク --- (tab4 に変更)
            with tab4:
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

                with st.expander("分析プロセスと論文記述例"):
                    st.markdown("""
                        #### 1. 分析プロセス
                        1.  **単語抽出**: WordCloudと同様に、名詞・動詞・形容詞からストップワードと数字を除外した単語リストを使用しました。
                        2.  **共起の定義**: 1つのドキュメント（Excelの1行）内で同時に出現した単語ペアを「共起」として定義しました。
                        3.  **頻度集計**: 全ドキュメントを対象に、共起する単語ペアの出現頻度を集計しました。
                        4.  **ネットワーク構築**: 共起頻度が高かった上位70ペアを抽出し、`NetworkX` ライブラリを用いてネットワークを構築しました。
                        5.  **ノードのサイズ**: 各単語の**全体での出現頻度**に基づき、ノード（円）のサイズを動的に変更しました。
                        6.  **ノードの色**: `greedy_modularity_communities` アルゴリズムを用い、関連性の高い単語グループ（コミュニティ）を検出し、グループごとに色分けしました。
                        7.  **エッジの太さ**: 共起頻度（関係の強さ）に基づき、エッジ（線）の太さを動的に変更しました（係数: 0.3）。
                        
                        #### 2. 論文記述例
                        > ...次に、単語間の関連性を探索するため、共起ネットワーク分析を実施した。共起頻度上位70ペアに基づきネットワーク（図2）を描画した。ノードのサイズは各単語の出現頻度を、エッジの太さは共起頻度を反映している。また、`greedy_modularity_communities` アルゴリズムによるコミュニティ検出を実行し、抽出されたクラスターごとに色分けを行った。
                        >
                        > 図2より、[単語A]や[単語B]が（大きなノードで示されるように）高頻度で出現していることがわかる。また、[単語C, D, E]が同じ色（[色]）のコミュニティを形成しており、これらが密接に関連するトピック群であることが示唆された。
                    """)

            # --- Tab 5: KWIC (文脈検索) --- (tab5 に変更)
            with tab5:
                st.subheader("KWIC (文脈検索)")
                st.info("キーワードに `*` を含めるとワイルドカード検索が可能です (例: `顧客*`)。")
                kwic_keyword = st.text_input("文脈を検索したい単語を入力してください", key="kwic_input")
                if kwic_keyword:
                    kwic_html_content = generate_kwic_html(df_analyzed, text_column, kwic_keyword)
                    html(kwic_html_content, height=400, scrolling=True)

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

            # --- Tab 6: 属性別 特徴語 --- (tab6 に変更)
            with tab6:
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

            # --- Tab 7: AI 学術論文 --- (tab7 に変更)
            with tab7:
                st.subheader("AIによる学術論文風サマリー")
                if 'ai_result_academic' not in st.session_state:
                    with st.spinner("AIによるクラスター分析と学術論文風サマリーを生成中..."):

                        if analyzed_items < total_items: st.warning(analysis_scope_warning, icon="⚠️")
                        else: st.info(analysis_scope_warning, icon="✅")
                        
                        try:
                            cluster_json_str = get_or_generate_cluster_json(ai_input_text, analysis_scope_instr, analyzed_items)
                        except Exception as e:
                            st.error(f"クラスターJSONの生成に失敗しました: {e}")
                            cluster_json_str = '{"name": "エラー", "children": []}'

                        contents_acad = [{"parts": [{"text": ai_input_text}]}]
                        has_attr_a = bool(attribute_columns)
                        has_attribute_str_a = "## 4. 属性間の比較分析 (Comparative Analysis)\n(属性（カテゴリ）間で見られた顕著な差異や特徴的な傾向について、具体的に比較・記述する)" if has_attr_a else ""
                        attr_instr_a = "データは「属性 || テキスト」の形式です。属性ごとの傾向や違いにも着目して分析してください。" if has_attr_a else ""
                        
                        system_instr_a = SYSTEM_PROMPT_ACADEMIC.format(
                            analysis_scope_instruction=analysis_scope_instr,
                            attributeInstruction=attr_instr_a, 
                            has_attribute=has_attribute_str_a,
                            cluster_json_data=cluster_json_str 
                        )
                        st.session_state.ai_result_academic = call_gemini_api(contents_acad, system_instruction=system_instr_a)
                st.markdown(st.session_state.ai_result_academic)
                
            # --- Tab 8: AI チャット --- (tab8 に変更)
            with tab8:
                st.subheader("💬 AI チャット (データ分析)")
                st.info("AIに質問できます。") 
                if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]): st.markdown(message["content"])
                if prompt := st.chat_input("AIに質問を入力してください (例: 主な課題は何ですか？)"):
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.spinner("AIが応答を生成中..."):
                        
                        if analyzed_items < total_items: 
                            with st.chat_message("assistant", avatar="⚠️"):
                                st.warning(f"（AIへの参照データは、全{total_items:,}件中、先頭{analyzed_items:,}件に制限されています）")
                        
                        context_text = ai_input_text

                        api_contents = []
                        first_user_message = f"""以下のテキストデータ（コンテキスト）について質問があります。\n\n--- コンテキスト ---\n{context_text}\n\n--- 質問 ---\n{prompt}"""
                        is_first_turn = len(st.session_state.chat_messages) == 1
                        
                        for msg in st.session_state.chat_messages[:-1]:
                            api_contents.append({"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]})
                        api_contents.append({"role": "user", "parts": [{"text": first_user_message if is_first_turn else prompt}]})

                        response = call_gemini_api(api_contents, system_instruction=SYSTEM_PROMPT_CHAT)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"): st.markdown(response)

    except Exception as e:
        st.error(f"ファイルの読み込みまたは分析中にエラーが発生しました: {e}")
        st.exception(e) # 詳細なエラーログを表示
