# ============================================================================
# Streamlit 앱: MAG 7 + BTC VWAP + Z-Score 분석 (Complete Enhanced Version)
# ============================================================================

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import json
from openai import OpenAI
import yfinance as yf
import time

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="MAG 7 + BTC Advanced Quant System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API 설정 및 초기화
# ============================================================================
GEMINI_ENABLED = False
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_ENABLED = True
except Exception as e:
    pass

OPENAI_ENABLED = False
OPENAI_CLIENT = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_CLIENT = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        OPENAI_ENABLED = True
except Exception as e:
    pass

# 모델 설정
OPENAI_MODEL_MARKET = st.secrets.get("OPENAI_MODEL_MARKET", "gpt-4o")
OPENAI_MODEL_STOCK  = st.secrets.get("OPENAI_MODEL_STOCK",  "gpt-4o-mini")
OPENAI_MODEL_CHAT   = st.secrets.get("OPENAI_MODEL_CHAT",   "gpt-4o")
GEMINI_MODEL_MARKET = "gemini-2.5-flash"
GEMINI_MODEL_STOCK  = "gemini-2.5-flash"

# ============================================================================
# 로그인 시스템
# ============================================================================
def check_password():
    if st.session_state.get('password_correct', False):
        return True
    
    st.title("🔒 MAG 7 + BTC Advanced Quant System")
    st.markdown("### Next-Gen AI-Powered Trading Analytics")
    
    with st.form("credentials"):
        username = st.text_input("아이디 (ID)", key="username")
        password = st.text_input("비밀번호 (Password)", type="password", key="password")
        submit_btn = st.form_submit_button("로그인", type="primary")
    
    if submit_btn:
        if username in st.secrets["passwords"] and password == st.secrets["passwords"][username]:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("😕 아이디 또는 비밀번호가 올바르지 않습니다.")
    return False

if not check_password():
    st.stop()

# ============================================================================
# 분석 클래스 (Enhanced Error Handling + Original Logic)
# ============================================================================
class MAG7BTCVWAPAnalyzer:
    def __init__(self, start_date='2020-01-01', end_date=None, burn_in_calendar_days=14):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.burn_in_calendar_days = burn_in_calendar_days
        self.results = {}
        self.errors = {}
        self.stocks = {
            'AAPL': 'Apple', 
            'MSFT': 'Microsoft', 
            'GOOGL': 'Alphabet',
            'AMZN': 'Amazon', 
            'NVDA': 'NVIDIA', 
            'META': 'Meta',
            'TSLA': 'Tesla', 
            'BTC-USD': 'Bitcoin', 
            'COIN': 'Coinbase'
        }

    def get_quarter_start_date(self, date):
        quarter = (date.month - 1) // 3 + 1
        start_month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
        return datetime(date.year, start_month, 1)

    def calculate_single(self, ticker, name):
        """개선된 단일 종목 분석 - 재시도 로직 포함"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"[{attempt+1}/{max_retries}] {name} ({ticker}) 데이터 다운로드 중...")
                
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=self.start_date, 
                    end=self.end_date, 
                    auto_adjust=False,
                    timeout=10
                )
                
                if df.empty:
                    raise ValueError(f"데이터 없음: {ticker}")
                
                print(f"✅ {name}: {len(df)}일 데이터 수집 완료")
                
                # 타임존 제거
                if df.index.tz is not None: 
                    df.index = df.index.tz_localize(None)
                
                # HLC3 계산
                df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
                df['Year'] = df.index.year
                df['Quarter'] = df.index.quarter
                df['YearQuarter'] = df['Year'].astype(str) + 'Q' + df['Quarter'].astype(str)
                
                # 초기화
                for col in ['Quarterly_VWAP', 'Quarterly_StdDev', 'Z_Score', 'Deviation_Amount']:
                    df[col] = 0.0
                df['Quarter_Start_Date'] = pd.NaT
                df['Is_Burn_In'] = False

                # 분기별 계산
                for quarter in df['YearQuarter'].unique():
                    quarter_mask = df['YearQuarter'] == quarter
                    quarter_data = df[quarter_mask].copy()
                    
                    if len(quarter_data) == 0: 
                        continue

                    first_date = quarter_data.index[0]
                    quarter_start = self.get_quarter_start_date(first_date)
                    burn_in_end_date = quarter_start + timedelta(days=self.burn_in_calendar_days)
                    
                    df.loc[quarter_mask, 'Quarter_Start_Date'] = quarter_start
                    
                    for idx in quarter_data.index:
                        df.loc[idx, 'Is_Burn_In'] = (idx < burn_in_end_date)
                    
                    # VWAP 계산
                    cumulative_tpv = (quarter_data['HLC3'] * quarter_data['Volume']).cumsum()
                    cumulative_volume = quarter_data['Volume'].cumsum()
                    quarter_vwap = cumulative_tpv / cumulative_volume.replace(0, np.nan)
                    
                    df.loc[quarter_mask, 'Quarterly_VWAP'] = quarter_vwap
                    # ⭐ 먼저 모든 Deviation_Amount 계산
                    df.loc[quarter_mask, 'Deviation_Amount'] = df.loc[quarter_mask, 'Close'] - quarter_vwap
                  
                    # StdDev & Z-Score
                    valid_mask = quarter_mask & (~df['Is_Burn_In'])
                    
                    if valid_mask.sum() > 1:
                        valid_deviations = df.loc[valid_mask, 'Deviation_Amount']
                        quarter_std = valid_deviations.std()
                        df.loc[quarter_mask, 'Quarterly_StdDev'] = quarter_std
                        
                        if quarter_std > 0 and not pd.isna(quarter_std):
                            df.loc[quarter_mask, 'Z_Score'] = df.loc[quarter_mask, 'Deviation_Amount'] / quarter_std
                        else:
                            df.loc[quarter_mask, 'Z_Score'] = 0
                    else:
                        # burn-in 제외 데이터가 1개 이하면 Z-Score 계산 불가
                        df.loc[quarter_mask, 'Quarterly_StdDev'] = 0
                        df.loc[quarter_mask, 'Z_Score'] = 0
                      

                # 추가 계산
                df['Deviation_Pct'] = (df['Deviation_Amount'] / df['Quarterly_VWAP']) * 100
                df['Below_VWAP'] = df['Close'] < df['Quarterly_VWAP']
                df['Above_VWAP'] = df['Close'] >= df['Quarterly_VWAP']
                
                # Z 구간
                bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
                labels = ['극단하방', '강한하방', '약한하방', '약한상방', '강한상방', '극단상방']
                df['Z_Zone'] = pd.cut(df['Z_Score'], bins=bins, labels=labels)

                # 유효 데이터
                df_valid = df[~df['Is_Burn_In']].copy()
                if df_valid.empty:
                    raise ValueError(f"유효 데이터 없음: {ticker}")

                current = df.iloc[-1]
                
                # 통계 계산
                below_days = df_valid[df_valid['Below_VWAP']]
                above_days = df_valid[df_valid['Above_VWAP']]
                total_days = len(df_valid)
                total_days_all = len(df)
                burn_in_days_count = total_days_all - total_days
                
                result = {
                    'ticker': ticker, 
                    'name': name, 
                    'df': df, 
                    'df_valid': df_valid,
                    'current_price': float(current['Close']),
                    'current_vwap': float(current['Quarterly_VWAP']),
                    'current_deviation': float(current['Deviation_Pct']),
                    'current_zscore': float(current['Z_Score']),
                    'current_zone': str(current['Z_Zone']),
                    'is_below_vwap': bool(current['Below_VWAP']),
                    'total_days': int(total_days),
                    'total_days_all': int(total_days_all),
                    'burn_in_days_count': int(burn_in_days_count),
                    'below_days_count': int(len(below_days)),
                    'below_days_pct': float((len(below_days) / total_days * 100) if total_days > 0 else 0),
                    'above_days_count': int(len(above_days)),
                    'above_days_pct': float((len(above_days) / total_days * 100) if total_days > 0 else 0),
                    'avg_deviation_all': float(df_valid['Deviation_Pct'].mean()),
                    'avg_deviation_below': float(below_days['Deviation_Pct'].mean()) if len(below_days) > 0 else 0.0,
                    'max_deviation_below': float(below_days['Deviation_Pct'].min()) if len(below_days) > 0 else 0.0,
                    'avg_deviation_above': float(above_days['Deviation_Pct'].mean()) if len(above_days) > 0 else 0.0,
                    'max_deviation_above': float(above_days['Deviation_Pct'].max()) if len(above_days) > 0 else 0.0,
                    'avg_zscore_all': float(df_valid['Z_Score'].mean()),
                    'avg_zscore_below': float(below_days['Z_Score'].mean()) if len(below_days) > 0 else 0.0,
                    'avg_zscore_above': float(above_days['Z_Score'].mean()) if len(above_days) > 0 else 0.0,
                    'min_zscore': float(df_valid['Z_Score'].min()),
                    'max_zscore': float(df_valid['Z_Score'].max()),
                    'min_zscore_below': float(below_days['Z_Score'].min()) if len(below_days) > 0 else 0.0,
                    'max_zscore_above': float(above_days['Z_Score'].max()) if len(above_days) > 0 else 0.0,
                    'zone_stats': {str(k): int(v) for k, v in df_valid.groupby('Z_Zone', observed=True).size().to_dict().items()}
                }
                
                return result
                
            except Exception as e:
                error_msg = f"시도 {attempt+1}/{max_retries} 실패: {str(e)}"
                print(f"❌ {name}: {error_msg}")
                
                if attempt < max_retries - 1:
                    print(f"⏳ {retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.errors[ticker] = error_msg
                    return None
        
        return None

    def analyze_all(self, max_workers=3):
        """병렬 분석 - 개선된 에러 핸들링"""
        success_count = 0
        total_count = len(self.stocks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.calculate_single, ticker, name): (ticker, name)
                for ticker, name in self.stocks.items()
            }
            
            for future in as_completed(futures):
                ticker, name = futures[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        self.results[ticker] = result
                        success_count += 1
                        print(f"✅ {name} 완료 ({success_count}/{total_count})")
                except Exception as e:
                    self.errors[ticker] = str(e)
                    print(f"❌ {name} 실패: {e}")
        
        return self

    def get_summary_table(self):
        """요약 테이블 생성"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for ticker, data in self.results.items():
            latest_date = data['df'].index[-1].strftime('%Y-%m-%d')
            
            if ticker == 'BTC-USD':
                price_format = f"${data['current_price']:,.2f}"
                vwap_format = f"${data['current_vwap']:,.2f}"
            else:
                price_format = f"${data['current_price']:,.2f}"
                vwap_format = f"${data['current_vwap']:,.2f}"
            
            summary_data.append({
                '순위': len(summary_data) + 1,
                '종목명': data['name'],
                '티커': ticker,
                '현재가': price_format,
                'VWAP': vwap_format,
                '현재괴리(%)': f"{data['current_deviation']:+.2f}",
                'VWAP상태': '🔴 아래' if data['is_below_vwap'] else '🟢 위',
                '전체평균괴리(%)': f"{data['avg_deviation_all']:+.2f}",
                '하방평균괴리(%)': f"{data['avg_deviation_below']:.2f}",
                '상방평균괴리(%)': f"{data['avg_deviation_above']:+.2f}",
                'VWAP아래비율(%)': f"{data['below_days_pct']:.1f}",
                'VWAP위비율(%)': f"{data['above_days_pct']:.1f}",
                '유효거래일': data['total_days'],
                '제외일': data['burn_in_days_count']
            })
        return pd.DataFrame(summary_data)

    def get_zscore_summary_table(self):
        """Z-Score 요약 테이블"""
        summary_data = []
        for ticker, data in self.results.items():
            if ticker == 'BTC-USD':
                price_format = f"${data['current_price']:,.2f}"
            else:
                price_format = f"${data['current_price']:,.2f}"
                
            summary_data.append({
                '순위': len(summary_data) + 1,
                '종목명': data['name'],
                '티커': ticker,
                '현재가': price_format,
                'Z-Score': f"{data['current_zscore']:+.2f}σ",
                'Z구간': str(data['current_zone']),
                '괴리(%)': f"{data['current_deviation']:+.2f}",
                '평균Z': f"{data['avg_zscore_all']:+.2f}σ",
                '최소Z': f"{data['min_zscore']:+.2f}σ",
                '최대Z': f"{data['max_zscore']:+.2f}σ",
                '하방평균Z': f"{data['avg_zscore_below']:+.2f}σ",
                '상방평균Z': f"{data['avg_zscore_above']:+.2f}σ"
            })
        return pd.DataFrame(summary_data)

    def get_trading_signals(self):
        """Z-Score 기반 매매 신호 테이블"""
        signals = []
        for ticker, data in self.results.items():
            z = data['current_zscore']

            if z <= -2:
                signal = '🟢 강력매수'
                reason = f'극단 저평가 (Z={z:.2f}σ, 역사적 최저 근접)'
            elif z <= -1:
                signal = '🟡 매수고려'
                reason = f'통계적 저평가 (Z={z:.2f}σ, 1σ 이하)'
            elif z >= 2:
                signal = '🔴 강력매도'
                reason = f'극단 고평가 (Z={z:.2f}σ, 역사적 최고 근접)'
            elif z >= 1:
                signal = '🟠 매도고려'
                reason = f'통계적 고평가 (Z={z:.2f}σ, 1σ 이상)'
            else:
                signal = '⚪ 중립'
                reason = f'정상 범위 (Z={z:.2f}σ)'

            if ticker == 'BTC-USD':
                price_format = f"${data['current_price']:,.2f}"
            else:
                price_format = f"${data['current_price']:,.2f}"

            signals.append({
                '종목명': data['name'],
                '티커': ticker,
                'Z-Score': f"{z:+.2f}σ",
                'Z구간': str(data['current_zone']),
                '신호': signal,
                '근거': reason,
                '괴리(%)': f"{data['current_deviation']:+.2f}",
                '현재가': price_format
            })

        df_signals = pd.DataFrame(signals)
        df_signals['Z_numeric'] = df_signals['Z-Score'].str.replace('σ', '').astype(float)
        df_signals = df_signals.sort_values('Z_numeric', ascending=True)
        df_signals = df_signals.drop('Z_numeric', axis=1)
        return df_signals

    def get_integrated_recommendations_table(self):
        """통합 추천 테이블"""
        recommendations = []
        for ticker, data in self.results.items():
            dev = data['current_deviation']
            z = data['current_zscore']

            if dev <= -5:
                if z <= -2:
                    signal = '🟢🟢 강력매수'
                    score = 5
                elif z <= -1:
                    signal = '🟡 매수고려'
                    score = 4
                else:
                    signal = '⚪ 변동성주의'
                    score = 3
            elif dev >= 5:
                if z >= 2:
                    signal = '🔴🔴 강력매도'
                    score = 1
                elif z >= 1:
                    signal = '🟠 매도고려'
                    score = 2
                else:
                    signal = '⚪ 중립'
                    score = 3
            else:
                signal = '⚪ 중립'
                score = 3

            if ticker == 'BTC-USD':
                price_format = f"${data['current_price']:,.2f}"
            else:
                price_format = f"${data['current_price']:,.2f}"

            recommendations.append({
                '점수': score,
                '종목명': data['name'],
                '티커': ticker,
                '통합신호': signal,
                '괴리(%)': f"{dev:+.2f}",
                'Z-Score': f"{z:+.2f}σ",
                '현재가': price_format,
                '하방여력(%)': f"{dev - data['max_deviation_below']:+.2f}"
            })

        df = pd.DataFrame(recommendations)
        df = df.sort_values('점수', ascending=False)
        df = df.drop('점수', axis=1)
        return df

    def plot_current_deviation_bar(self):
        """현재 괴리율 막대 차트"""
        data_list = []
        for ticker, data in self.results.items():
            data_list.append({
                'name': data['name'],
                'deviation': data['current_deviation'],
                'avg_all': data['avg_deviation_all'],
                'is_below': data['is_below_vwap']
            })

        df_plot = pd.DataFrame(data_list)
        df_plot = df_plot.sort_values('deviation')
        colors = ['red' if below else 'green' for below in df_plot['is_below']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_plot['name'],
            x=df_plot['deviation'],
            orientation='h',
            marker_color=colors,
            text=df_plot.apply(lambda x: f"{x['deviation']:+.2f}% (평:{x['avg_all']:+.1f}%)", axis=1),
            textposition='outside'
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="blue", line_width=2)
        fig.add_vline(x=-5, line_dash="dot", line_color="orange", opacity=0.5)
        fig.add_vline(x=-10, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_vline(x=5, line_dash="dot", line_color="lightgreen", opacity=0.5)

        fig.update_layout(
            title=f"MAG 7 + BTC 현재 VWAP 괴리율",
            xaxis_title="괴리율 (%)",
            yaxis_title="종목",
            height=500,
            showlegend=False
        )

        return fig

    def plot_zscore_ranking(self):
        """Z-Score 순위 차트"""
        data_list = []
        for ticker, data in self.results.items():
            data_list.append({
                'name': data['name'],
                'zscore': data['current_zscore'],
                'deviation': data['current_deviation']
            })

        df_plot = pd.DataFrame(data_list)
        df_plot = df_plot.sort_values('zscore')

        colors = ['darkred' if z <= -2 else 'red' if z <= -1 else 'gray' if z < 1 else 'orange' if z < 2 else 'darkgreen'
                  for z in df_plot['zscore']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_plot['name'],
            x=df_plot['zscore'],
            orientation='h',
            marker_color=colors,
            text=df_plot.apply(lambda x: f"Z={x['zscore']:+.2f}σ ({x['deviation']:+.1f}%)", axis=1),
            textposition='outside'
        ))

        for z_val, color in [(0, 'blue'), (-1, 'orange'), (-2, 'red'), (1, 'lightgreen'), (2, 'green')]:
            fig.add_vline(x=z_val, line_dash="dash", line_color=color, line_width=2, opacity=0.7)

        fig.update_layout(
            title="MAG 7 + BTC Z-Score 순위",
            xaxis_title="Z-Score (표준편차)",
            yaxis_title="종목",
            height=500,
            showlegend=False
        )

        return fig

    def plot_price_vwap_zscore_interactive(self, ticker):
        """개별 종목 상세 차트"""
        if ticker not in self.results:
            return None

        data = self.results[ticker]
        df = data['df']

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f"{data['name']} - 가격 vs VWAP",
                "VWAP 괴리율 (%)",
                "Z-Score (표준편차)"
            )
        )

        # 가격 vs VWAP
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='종가', line=dict(color='black', width=1.5)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['Quarterly_VWAP'], name='VWAP', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # VWAP 아래 표시
        below_mask = df['Below_VWAP']
        fig.add_trace(
            go.Scatter(
                x=df[below_mask].index, y=df[below_mask]['Close'],
                mode='markers', name='VWAP 아래',
                marker=dict(color='red', size=3, opacity=0.5)
            ),
            row=1, col=1
        )

        # 괴리율
        colors = ['red' if x < 0 else 'green' for x in df['Deviation_Pct']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Deviation_Pct'], name='괴리율', marker_color=colors, opacity=0.6),
            row=2, col=1
        )

        # Z-Score
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Z_Score'], name='Z-Score',
                mode='lines', line=dict(color='purple', width=1.5),
                fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)'
            ),
            row=3, col=1
        )

        # 기준선
        for z_val, color in [(0, 'blue'), (-1, 'orange'), (-2, 'red'), (1, 'lightgreen'), (2, 'green')]:
            fig.add_hline(y=z_val, line_dash="dash", line_color=color, opacity=0.5, row=3, col=1)

        fig.update_layout(
            height=1000,
            showlegend=True,
            hovermode='x unified',
            title_text=f"{data['name']} - 괴리: {data['current_deviation']:+.2f}% | Z-Score: {data['current_zscore']:+.2f}σ"
        )

        fig.update_xaxes(title_text="날짜", row=3, col=1)
        fig.update_yaxes(title_text="가격 ($)", row=1, col=1)
        fig.update_yaxes(title_text="괴리율 (%)", row=2, col=1)
        fig.update_yaxes(title_text="Z-Score (σ)", row=3, col=1)

        return fig

# ============================================================================
# Enhanced Dual AI Handler
# ============================================================================
class EnhancedDualAIHandler:
    @staticmethod
    def generate_market_context(analyzer):
        if not analyzer or not analyzer.results:
            return "현재 분석된 데이터가 없습니다."
        
        context = f"### 📊 시장 분석 데이터 (생성: {datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
        
        all_z = [d['current_zscore'] for d in analyzer.results.values()]
        context += f"**시장 전반:**\n"
        context += f"- 평균 Z-Score: {np.mean(all_z):.2f}σ\n"
        context += f"- Z-Score 범위: {np.min(all_z):.2f}σ ~ {np.max(all_z):.2f}σ\n"
        context += f"- 극단저평가(Z≤-2): {sum(1 for z in all_z if z <= -2)}개\n"
        context += f"- 극단고평가(Z≥2): {sum(1 for z in all_z if z >= 2)}개\n\n"
        
        context += "**개별 종목:**\n"
        for ticker, data in analyzer.results.items():
            context += f"- **{data['name']} ({ticker})**\n"
            context += f"  현재가: ${data['current_price']:,.2f} | VWAP: ${data['current_vwap']:,.2f}\n"
            context += f"  괴리: {data['current_deviation']:+.2f}% | Z: {data['current_zscore']:.2f}σ ({data['current_zone']})\n"
            
        return context

    @staticmethod
    def generate_stock_context(ticker, data):
        context = f"### 🔍 {data['name']} ({ticker}) 상세 분석\n\n"
        context += f"**현재 상태 ({datetime.now().strftime('%Y-%m-%d')})**\n"
        context += f"- 현재가: ${data['current_price']:,.2f}\n"
        context += f"- VWAP: ${data['current_vwap']:,.2f}\n"
        context += f"- 괴리율: {data['current_deviation']:+.2f}%\n"
        context += f"- Z-Score: {data['current_zscore']:.2f}σ\n"
        context += f"- Z구간: {data['current_zone']}\n"
        context += f"- VWAP 대비: {'저평가(Below)' if data['is_below_vwap'] else '고평가(Above)'}\n\n"
        
        context += f"**역사적 통계 (유효거래일: {data['total_days']}일)**\n"
        context += f"- 평균 괴리: {data['avg_deviation_all']:+.2f}%\n"
        context += f"- 하방 평균: {data['avg_deviation_below']:.2f}% (최대: {data['max_deviation_below']:.2f}%)\n"
        context += f"- 상방 평균: {data['avg_deviation_above']:+.2f}% (최대: {data['max_deviation_above']:+.2f}%)\n"
        context += f"- 평균 Z: {data['avg_zscore_all']:.2f}σ\n"
        context += f"- Z 범위: {data['min_zscore']:.2f}σ ~ {data['max_zscore']:.2f}σ\n"
        
        return context

    @staticmethod
    def gemini_market_analysis(context):
        if not GEMINI_ENABLED:
            return "⚠️ Gemini API가 비활성화되어 있습니다."
        
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_MARKET)
            prompt = f"""
당신은 월스트리트 시니어 퀀트 애널리스트입니다.

{context}

위 데이터를 바탕으로 심층 시장 분석 리포트를 작성하세요:

1. 시장 전반 진단
2. 매수/매도 우선순위
3. 리스크 분석
4. 실행 전략

**길이:** 1000-1500단어
**언어:** 한국어
"""
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Gemini 오류: {str(e)}"

    @staticmethod
    def gemini_stock_analysis(ticker, context):
        if not GEMINI_ENABLED:
            return "⚠️ Gemini API가 비활성화되어 있습니다."
        
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_STOCK)
            prompt = f"""
{context}

위 종목의 실전 트레이딩 가이드를 작성하세요:
1. 현재 위치 해석
2. 평균회귀 전략
3. 추세 추종 전략
4. 리스크 관리
5. 실행 체크리스트

**언어:** 한국어
"""
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Gemini 오류: {str(e)}"

    @staticmethod
    def openai_market_analysis(context):
        if not OPENAI_ENABLED:
            return "⚠️ OpenAI API가 비활성화되어 있습니다."
        
        try:
            prompt = f"""
{context}

정량적 시장 분석을 제공하세요:
1. 시장 진단
2. 매수/매도 우선순위
3. 리스크
4. 실행 플랜
"""
            messages = [
                {"role": "system", "content": "너는 퀀트 애널리스트다."},
                {"role": "user", "content": prompt}
            ]
            
            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL_MARKET,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ OpenAI 오류: {str(e)}"

    @staticmethod
    def openai_stock_analysis(ticker, context):
        if not OPENAI_ENABLED:
            return "⚠️ OpenAI API가 비활성화되어 있습니다."
        
        try:
            prompt = f"""
{context}

실행 가능한 트레이딩 플랜을 제시하세요.
"""
            messages = [
                {"role": "system", "content": "너는 퀀트 트레이더다."},
                {"role": "user", "content": prompt}
            ]
            
            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL_STOCK,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ OpenAI 오류: {str(e)}"

    @staticmethod
    def query_advanced_chat(prompt, context, model_choice, chat_history):
        """
        Advanced Chat: 시장 데이터(Context)와 대화 히스토리를 결합하여 
        AI가 현재 상황을 인지한 상태로 답변하도록 유도
        """
        # 1. 강력한 페르소나 및 데이터 주입
        system_instruction = f"""
        당신은 월스트리트의 시니어 퀀트 트레이더이자 데이터 분석가입니다.
        
        [현재 실시간 시장 분석 데이터]
        {context}
        
        [지시사항]
        1. 위 [시장 분석 데이터]에 있는 수치(Z-Score, 괴리율 등)를 근거로 답변하세요.
        2. 사용자의 질문이 데이터와 관련 없다면 일반적인 금융 지식으로 답하세요.
        3. 감정적인 희망 회로보다는, 통계적 수치에 기반한 객관적인 뷰를 제시하세요.
        4. 한국어로 간결하고 명확하게 답변하세요.
        """

        # 2. Gemini 로직 (긴 컨텍스트 처리에 강함 -> 텍스트 합치기 방식)
        if model_choice == "Gemini":
            if not GEMINI_ENABLED: return "⚠️ Gemini API 설정이 필요합니다."
            try:
                model = genai.GenerativeModel("gemini-2.5-flash") 
                
                full_prompt = system_instruction + "\n\n[이전 대화 내역]\n"
                for msg in chat_history[-10:]: # 최근 10개 대화 기억
                    role_label = "User" if msg['role'] == 'user' else "AI"
                    full_prompt += f"{role_label}: {msg['content']}\n"
                
                full_prompt += f"\n[User 질문]: {prompt}\n[AI 답변]:"
                
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                return f"⚠️ Gemini 오류: {str(e)}"

        # 3. OpenAI 로직 (System Message 구조 활용)
        else: 
            if not OPENAI_ENABLED: return "⚠️ OpenAI API 설정이 필요합니다."
            try:
                messages = [{"role": "system", "content": system_instruction}]
                messages.extend(chat_history[-6:]) # 최근 6턴 기억
                messages.append({"role": "user", "content": prompt})
                
                response = OPENAI_CLIENT.chat.completions.create(
                    model=st.secrets.get("OPENAI_MODEL_CHAT", "gpt-4o"),
                    messages=messages,
                    temperature=0.3 # 퀀트 분석이므로 창의성보다는 논리성 중시
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"⚠️ OpenAI 오류: {str(e)}"
                
# ============================================================================
# 메인 앱
# ============================================================================
def main():
    # 사이드바
    with st.sidebar:
        st.header("⚙️ System Control")
        
        col1, col2 = st.columns(2)
        with col1:
            if GEMINI_ENABLED:
                st.success("✅ Gemini")
            else:
                st.error("❌ Gemini")
        with col2:
            if OPENAI_ENABLED:
                st.success("✅ OpenAI")
            else:
                st.error("❌ OpenAI")
        
        st.markdown("---")
        
        st.subheader("📊 분석 설정")
        burn_in = st.slider("Burn-in Period (일)", 7, 30, 14)
        
        period_options = {
            "최근 1년": 365,
            "최근 2년": 730,
            "최근 3년": 1095,
            "최근 5년": 1825,
            "2020년 이후": "2020-01-01",
            "2015년 이후": "2015-01-01"
        }
        
        period_choice = st.selectbox("분석 기간", list(period_options.keys()), index=3)
        
        if st.button("🚀 데이터 분석 실행", type="primary", use_container_width=True):
            period_val = period_options[period_choice]
            if isinstance(period_val, int):
                start_date = (datetime.now() - timedelta(days=period_val)).strftime('%Y-%m-%d')
            else:
                start_date = period_val
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📊 데이터 수집 중...")
            analyzer = MAG7BTCVWAPAnalyzer(start_date=start_date, burn_in_calendar_days=burn_in)
            
            progress_bar.progress(30)
            status_text.text("🔄 분석 실행 중... (병렬 처리)")
            analyzer.analyze_all(max_workers=3)
            
            progress_bar.progress(80)
            
            if analyzer.results:
                st.session_state['analyzer'] = analyzer
                st.session_state['market_context'] = EnhancedDualAIHandler.generate_market_context(analyzer)
                st.session_state['analysis_time'] = datetime.now()
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ 분석 완료! ({len(analyzer.results)}/{len(analyzer.stocks)}개 종목)")
                
                if analyzer.errors:
                    with st.expander("⚠️ 실패한 종목"):
                        for ticker, error in analyzer.errors.items():
                            st.error(f"**{analyzer.stocks[ticker]} ({ticker})**: {error}")
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ 모든 종목 분석 실패. 네트워크 연결을 확인하세요.")
        
        if st.button("🚪 로그아웃"):
            st.session_state['password_correct'] = False
            st.rerun()

    st.title("🧬 MAG 7 + BTC Advanced Quant System")
    st.markdown("##### AI-Powered Statistical Arbitrage Platform")
    
    current_datetime = datetime.now()
    st.markdown(f"**📅 분석 생성 일시:** {current_datetime.strftime('%Y년 %m월 %d일 %H:%M:%S')} (KST)")
    
    if 'analyzer' in st.session_state:
        analyzer = st.session_state['analyzer']
        
        if analyzer.results:
            first_ticker = list(analyzer.results.keys())[0]
            data_start = analyzer.results[first_ticker]['df'].index[0].strftime('%Y-%m-%d')
            data_end = analyzer.results[first_ticker]['df'].index[-1].strftime('%Y-%m-%d')
            st.markdown(f"**📊 데이터 기간:** {data_start} ~ {data_end}")
    
    st.markdown("---")
    
    if 'analyzer' in st.session_state:
        analyzer = st.session_state['analyzer']
        
        if not analyzer.results:
            st.error("⚠️ 분석된 데이터가 없습니다.")
            return
        
        tabs = st.tabs([
            "📊 요약",
            "📈 VWAP 분석",
            "🎯 Z-Score 분석",
            "🤖 Gemini 분석",
            "🧠 OpenAI 분석",
            "💡 통합 신호",
            "🔍 개별 종목",
            "💬 AI 채팅"
        ])
        
        # 탭 1: 요약
        with tabs[0]:
            st.header("📊 종합 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            
            all_zscores = [data['current_zscore'] for data in analyzer.results.values()]
            avg_z = np.mean(all_zscores)
            
            extreme_low = sum(1 for z in all_zscores if z <= -2)
            extreme_high = sum(1 for z in all_zscores if z >= 2)
            
            below_vwap_count = sum(1 for data in analyzer.results.values() if data['is_below_vwap'])
            
            with col1:
                st.metric("평균 Z-Score", f"{avg_z:+.2f}σ")
            with col2:
                st.metric("극단저평가", f"{extreme_low}개", help="Z-Score ≤ -2σ")
            with col3:
                st.metric("극단고평가", f"{extreme_high}개", help="Z-Score ≥ 2σ")
            with col4:
                st.metric("VWAP 아래", f"{below_vwap_count}개")
            
            st.markdown("---")
            
            summary_df = analyzer.get_summary_table()
            st.dataframe(summary_df, use_container_width=True, height=400)
        
        # 탭 2: VWAP 분석
        with tabs[1]:
            st.header("📈 VWAP 괴리율 분석")
            
            fig_deviation = analyzer.plot_current_deviation_bar()
            st.plotly_chart(fig_deviation, use_container_width=True)
        
        # 탭 3: Z-Score 분석
        with tabs[2]:
            st.header("🎯 Z-Score 통계 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Z-Score 요약")
                zscore_summary = analyzer.get_zscore_summary_table()
                st.dataframe(zscore_summary, use_container_width=True, height=350)
            
            with col2:
                st.subheader("💡 매매 신호")
                signals = analyzer.get_trading_signals()
                st.dataframe(signals, use_container_width=True, height=350)
            
            st.markdown("---")
            
            fig_zscore = analyzer.plot_zscore_ranking()
            st.plotly_chart(fig_zscore, use_container_width=True)
        
        # 탭 4: Gemini 분석
        with tabs[3]:
            st.header("🤖 Gemini AI 심층 분석")
            
            if not GEMINI_ENABLED:
                st.error("❌ Gemini AI가 비활성화되어 있습니다.")
                st.info("secrets.toml에 GEMINI_API_KEY를 추가하세요.")
            else:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📌 Gemini 종합 분석")
                    if st.button("🚀 종합 분석 실행", type="primary", key="gemini_market"):
                        with st.spinner("🤖 Gemini 분석 중..."):
                            context = st.session_state.get('market_context', '')
                            analysis = EnhancedDualAIHandler.gemini_market_analysis(context)
                            st.session_state['gemini_market_report'] = analysis
                    
                    if 'gemini_market_report' in st.session_state:
                        st.markdown("### 📝 종합 리포트")
                        st.markdown(st.session_state['gemini_market_report'])
                        st.download_button(
                            "📥 다운로드 (TXT)",
                            data=st.session_state['gemini_market_report'],
                            file_name=f"Gemini_Market_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    st.subheader("🔍 Gemini 개별 종목 분석")
                    stock_names = [data['name'] for data in analyzer.results.values()]
                    selected_stock = st.selectbox("종목 선택", stock_names, key="gemini_stock_select")
                    
                    selected_ticker = None
                    for ticker, data in analyzer.results.items():
                        if data['name'] == selected_stock:
                            selected_ticker = ticker
                            break
                    
                    if st.button("🧠 종목 분석 실행", key="gemini_stock"):
                        with st.spinner("🤖 종목 분석 중..."):
                            if selected_ticker:
                                context = EnhancedDualAIHandler.generate_stock_context(
                                    selected_ticker, 
                                    analyzer.results[selected_ticker]
                                )
                                analysis = EnhancedDualAIHandler.gemini_stock_analysis(selected_ticker, context)
                                st.session_state['gemini_stock_report'] = analysis
                    
                    if 'gemini_stock_report' in st.session_state:
                        st.markdown("### 🧾 종목 리포트")
                        st.markdown(st.session_state['gemini_stock_report'])
        
        # 탭 5: OpenAI 분석
        with tabs[4]:
            st.header("🧠 OpenAI 종합/개별 분석")
            
            if not OPENAI_ENABLED:
                st.error("❌ OpenAI가 비활성화되어 있습니다.")
            else:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📌 OpenAI 종합 분석")
                    if st.button("🚀 종합 분석 실행", type="primary", key="openai_market"):
                        with st.spinner("🧠 OpenAI 분석 중..."):
                            context = st.session_state.get('market_context', '')
                            analysis = EnhancedDualAIHandler.openai_market_analysis(context)
                            st.session_state['openai_market_report'] = analysis
                    
                    if 'openai_market_report' in st.session_state:
                        st.markdown("### 📝 종합 리포트")
                        st.markdown(st.session_state['openai_market_report'])
                        st.download_button(
                            "📥 다운로드 (TXT)",
                            data=st.session_state['openai_market_report'],
                            file_name=f"OpenAI_Market_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    st.subheader("🔍 OpenAI 개별 종목 분석")
                    stock_names = [data['name'] for data in analyzer.results.values()]
                    selected_stock = st.selectbox("종목 선택", stock_names, key="openai_stock_select")
                    
                    selected_ticker = None
                    for ticker, data in analyzer.results.items():
                        if data['name'] == selected_stock:
                            selected_ticker = ticker
                            break
                    
                    if st.button("🧠 종목 분석 실행", key="openai_stock"):
                        with st.spinner("🧠 종목 분석 중..."):
                            if selected_ticker:
                                context = EnhancedDualAIHandler.generate_stock_context(
                                    selected_ticker,
                                    analyzer.results[selected_ticker]
                                )
                                analysis = EnhancedDualAIHandler.openai_stock_analysis(selected_ticker, context)
                                st.session_state['openai_stock_report'] = analysis
                    
                    if 'openai_stock_report' in st.session_state:
                        st.markdown("### 🧾 종목 리포트")
                        st.markdown(st.session_state['openai_stock_report'])
        
        # 탭 6: 통합 신호
        with tabs[5]:
            st.header("💡 통합 추천 시스템")
            st.markdown("**괴리율 + Z-Score 기반 통합 매매 신호**")
            
            integrated_table = analyzer.get_integrated_recommendations_table()
            st.dataframe(integrated_table, use_container_width=True, height=400)
            
            st.markdown("---")
            st.info("""
            **신호 해석 가이드:**
            - 🟢🟢 강력매수: 괴리 ≤ -5% AND Z-Score ≤ -2σ
            - 🟡 매수고려: 괴리 ≤ -5% AND Z-Score -2σ ~ -1σ
            - 🔴🔴 강력매도: 괴리 ≥ 5% AND Z-Score ≥ 2σ
            - 🟠 매도고려: 괴리 ≥ 5% AND Z-Score 1σ ~ 2σ
            - ⚪ 중립/변동성주의: 기타 경우
            """)
        
        # 탭 7: 개별 종목
        with tabs[6]:
            st.header("🔍 개별 종목 상세 분석")
            
            stock_names = [data['name'] for data in analyzer.results.values()]
            selected_stock = st.selectbox("종목 선택", stock_names, key="detail_stock_select")
            
            selected_ticker = None
            for ticker, data in analyzer.results.items():
                if data['name'] == selected_stock:
                    selected_ticker = ticker
                    break
            
            if selected_ticker:
                fig_detail = analyzer.plot_price_vwap_zscore_interactive(selected_ticker)
                if fig_detail:
                    st.plotly_chart(fig_detail, use_container_width=True)
                
                data = analyzer.results[selected_ticker]
                
                col1, col2, col3, col4 = st.columns(4)
                
                if selected_ticker == 'BTC-USD':
                    price_format = f"${data['current_price']:,.2f}"
                    vwap_format = f"${data['current_vwap']:,.2f}"
                else:
                    price_format = f"${data['current_price']:,.2f}"
                    vwap_format = f"${data['current_vwap']:,.2f}"
                
                with col1:
                    st.metric("현재가", price_format)
                    st.metric("VWAP", vwap_format)
                
                with col2:
                    st.metric("괴리율", f"{data['current_deviation']:+.2f}%")
                    st.metric("평균 괴리", f"{data['avg_deviation_all']:+.2f}%")
                
                with col3:
                    st.metric("Z-Score", f"{data['current_zscore']:+.2f}σ")
                    st.metric("Z 구간", str(data['current_zone']))
                
                with col4:
                    st.metric("유효 거래일", f"{data['total_days']}일")
                    st.metric("Burn-in 제외", f"{data['burn_in_days_count']}일")
        
        # 탭 8: AI 채팅
        # 탭 8: Advanced AI 채팅
        with tabs[7]:
            st.header("💬 Advanced Quant Chatbot")
            
            # 레이아웃: 채팅창(왼쪽) vs 제어패널(오른쪽)
            col_chat, col_ctrl = st.columns([3, 1])
            
            # 1. 오른쪽 제어 패널
            with col_ctrl:
                st.markdown("### 🎛️ 제어 패널")
                
                available_models = []
                if OPENAI_ENABLED: available_models.append("OpenAI")
                if GEMINI_ENABLED: available_models.append("Gemini")
                
                if not available_models:
                    st.error("API 키가 없습니다.")
                    model_choice = None
                else:
                    model_choice = st.radio("🧠 모델 선택", available_models, index=0)
                
                st.info(f"**모드 특징**\n- Gemini: 거시경제/종합해석\n- OpenAI: 수치분석/논리")
                
                st.markdown("---")
                if st.button("🧹 대화 지우기", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()
                
                with st.expander("데이터 컨텍스트 확인"):
                    st.caption(st.session_state.get('market_context', '데이터 분석을 먼저 실행하세요.'))

            # 2. 왼쪽 채팅창
            with col_chat:
                # 초기화
                if "chat_messages" not in st.session_state:
                    st.session_state.chat_messages = []

                # 대화 기록 표시
                for msg in st.session_state.chat_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # ✨ [추가된 기능] 빠른 질문 버튼 (Quick Replies)
                # 사용자가 자주 물어볼만한 핵심 질문 5가지 정의
                quick_questions = [
                    "📉 가장 저평가된(Z<-2) 종목은?",
                    "₿ 현재 비트코인 상태 분석해줘",
                    "⚠️ 지금 조심해야 할(과열) 종목은?",
                    "📊 전체 시장 분위기 한마디로 요약해",
                    "💡 오늘 추천하는 매매 전략은?"
                ]
                
                # 버튼을 가로로 배열
                btn_cols = st.columns(len(quick_questions))
                triggered_prompt = None
                
                for i, question in enumerate(quick_questions):
                    # 버튼 클릭 시 해당 질문을 저장
                    if btn_cols[i].button(question, key=f"quick_btn_{i}", use_container_width=True):
                        triggered_prompt = question

                # 3. 입력 처리 (채팅창 입력 OR 버튼 클릭)
                user_input = st.chat_input("질문을 입력하세요 (예: NVDA Z가 -1.5면 어떻게 할까?)")
                
                # 버튼이 눌렸거나, 채팅창에 입력이 들어오면 실행
                final_prompt = triggered_prompt if triggered_prompt else user_input

                if final_prompt:
                    if not model_choice:
                        st.error("AI 모델을 선택해주세요.")
                    else:
                        # 사용자 메시지 표시 및 저장
                        st.chat_message("user").markdown(final_prompt)
                        st.session_state.chat_messages.append({"role": "user", "content": final_prompt})

                        # AI 응답 생성
                        with st.chat_message("assistant"):
                            with st.spinner(f"🧠 {model_choice}가 퀀트 데이터를 분석 중입니다..."):
                                context = st.session_state.get('market_context', "")
                                
                                response = EnhancedDualAIHandler.query_advanced_chat(
                                    prompt=final_prompt,
                                    context=context,
                                    model_choice=model_choice,
                                    chat_history=st.session_state.chat_messages
                                )
                                
                                st.markdown(response)
                                st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        
                
           
    else:
        st.info("👈 사이드바에서 **'데이터 분석 실행'** 버튼을 눌러 시작하세요.")

if __name__ == "__main__":
    main()
