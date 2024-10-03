import FinanceDataReader as fdr
import mplfinance as mpf
import os

# 2015년부터 2024년까지의 데이터 가져오기 (S&P 500 ETF: SPY, IVV, VOO, QQQ, DIA)
symbols = ['SPY', 'IVV', 'VOO', 'QQQ', 'DIA']

# 각 종목의 데이터를 저장할 딕셔너리
data_dict = {}

for symbol in symbols:
    data_dict[symbol] = fdr.DataReader(symbol, '1980', '2024')

# 저장할 디렉토리 설정
if not os.path.exists('상승'):
    os.makedirs('상승')

if not os.path.exists('하락'):
    os.makedirs('하락')

# 종목별로 데이터를 봉 차트로 플로팅하고 저장하는 함수
def plot_save_candlestick(data, symbol, folder_name, date):
    # mplfinance를 사용하여 봉 차트 그리기
    mpf.plot(data, type='candle', volume=False,
             ylabel='Price', xlabel='Date',
             savefig=f'{folder_name}/{symbol}_candlestick_chart_{date}.png')

# 특정일의 가격이 상승 또는 하락하는 경우 이전 30일 차트를 그리는 로직
for symbol in symbols:
    print(f"\n[작업 시작] {symbol} 종목 처리 중...")

    for i in range(30, len(data_dict[symbol])):  # 30일부터 시작
        # 특정일의 종가와 이전날 종가 비교
        if data_dict[symbol]['Close'].iloc[i] > data_dict[symbol]['Close'].iloc[i - 1]:
            # 상승하는 경우: 이전 30일의 데이터를 가져온다
            chunk = data_dict[symbol].iloc[i - 30:i]
            plot_save_candlestick(chunk, symbol, '상승', data_dict[symbol].index[i].strftime('%Y-%m-%d'))
            print(f"[{symbol}] {data_dict[symbol].index[i].strftime('%Y-%m-%d')} 상승 봉 차트 저장 완료.")

        elif data_dict[symbol]['Close'].iloc[i] < data_dict[symbol]['Close'].iloc[i - 1]:
            # 하락하는 경우: 이전 30일의 데이터를 가져온다
            chunk = data_dict[symbol].iloc[i - 30:i]
            plot_save_candlestick(chunk, symbol, '하락', data_dict[symbol].index[i].strftime('%Y-%m-%d'))
            print(f"[{symbol}] {data_dict[symbol].index[i].strftime('%Y-%m-%d')} 하락 봉 차트 저장 완료.")

    print(f"[작업 완료] {symbol} 종목 처리 완료.")