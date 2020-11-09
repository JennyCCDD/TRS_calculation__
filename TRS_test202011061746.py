# -*- coding: utf-8 -*-
"""
@author: Mengxuan Chen
@emails: CHENMENGXUAN163@pingan.com.cn
@description:
    # 互换收益计算
@revise log:
    2020.11.03 创建程序
    2020.11.04 平均持仓成本用FIFO法计算
                股票和股指期货分别计算
                封装函数
    2020.11.05 股票和股指期货合并计算
                计算固定端收益
                整体修复日期和index的bug
                修复FIFO
                测试：
    2020.11.06 测试并修改FIFO
               股指期货价格adj=股指期货价格*合约乘数
               修复当天未交易当时有持仓的浮盈计算
               修复当天交易所有持仓股票的情况
               单独处理正好完全清仓的情况
"""
# In[]
import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import warnings

warnings.filterwarnings('ignore')
from WindPy import *
w.start()


def extractData(Date, Codes, type='stock'):
    '''
    使用wind API 接口提取数据，股票对应的是收盘价数据，股指期货对应结算价数据
    :param Date:
    :param Codes:
    :return:
    '''
    codes = ','.join(Codes)
    if isinstance(Date, str) == True:
        pass
    else:
        Date = datetime.strftime(Date, "%Y-%m-%d")
    if type == 'stock':
        tradeDate = "tradeDate=" + str(Date).replace('-', '') + ";priceAdj=U;cycle=D"
        data = w.wss(codes, "close", tradeDate)
    elif type == 'futures':
        tradeDate = "tradeDate=" + str(Date).replace('-', '') + ";cycle=D"
        data = w.wss(codes, "settle", tradeDate)
    else:
        raise ValueError('we do not have this type of product!')
    name = w.wss(codes, "sec_name", tradeDate)
    df1 = pd.DataFrame(name.Data[0], columns=name.Fields, index=name.Codes)
    df2 = pd.DataFrame(data.Data[0], columns=data.Fields, index=data.Codes)
    df = pd.concat([df1, df2], axis=1)
    df.reset_index(inplace=True)
    df.columns = ['证券代码', '证券简称', '收盘价']
    return df

def extractContractmultiplier(Codes):
    '''
    # 提取期货合约的合约乘数
    :param Date:
    :param Codes:
    :return:
    '''
    codes = ','.join(Codes)
    data = w.wss(codes, "contractmultiplier")
    df = pd.DataFrame(data.Data[0], columns=data.Fields, index=data.Codes)
    return df

def nameNorm(code, type):
    '''
    对每天输入数据的股票代码进行标准化处理
    :param code:
    :param type:
    :return:
    '''
    if type == 'stock':
        code = str(int(code))
        if len(code) < 6:
            code = (6 - len(code)) * '0' + code
        else:
            pass
        if code[0] == '6':
            code = code + '.SH'
        elif code[0:3] == '688':
            code = code + '.SZ'
        else:
            code = code + '.SZ'
    elif type == 'futures':
        code = str(code)
        code = code + '.CFE'
    else:
        raise ValueError('we do not have this type of product!')
    return code


def FIFO(df):
    '''
    # 使用FIFO方法计算成本价格
    :param df:
    :return:
    '''
    if len(df.loc[(df['是否卖出'] == False) & (df['剩余数量'] > 0)]) == len(df.loc[(df['是否卖出'] == False)]):
        df.loc[(df['是否卖出'] == False) & (df['剩余数量'] > 0),'剩余数量'] = \
            df.loc[(df['是否卖出'] == False) & (df['剩余数量'] > 0),'交易数量']
    dff = df.loc[(df['剩余数量'] > 0)].append(df.iloc[-1])
    # df = df.loc[~df['是否卖出'] == True]
    sum = 0
    xx = 0
    for kk, ii in enumerate(list(dff['交易数量'])):
        if sum > abs(dff['交易数量'].iloc[-1]):
            xx = kk
            break
        sum += ii
    if float(xx) > 0:
        df_x = dff.iloc[0:int(xx)]
        a = (df_x['交易价格'].iloc[:-1] * df_x['剩余数量'].iloc[:-1] ).sum()
        b = df_x['交易价格'].iloc[-1] * (abs(df['交易数量'].iloc[-1]) - df_x['剩余数量'].iloc[:-1].sum())
        c = abs(df['交易数量'].iloc[-1])
        df['成本价格'].iloc[-1] = (a+b) / c
        # 先计算最近一笔的剩余数量，再将之前吃掉的所有单的剩余数量设为0
        d = (abs(df['交易数量'].iloc[-1]) - df_x['剩余数量'].iloc[:-1].sum())
        df['剩余数量'].iloc[0:int(xx)][-1:] = df_x['剩余数量'].iloc[-1] - d
        df['剩余数量'].iloc[0:int(xx)][:-1] = 0
    else:
        # 正好完全清仓的情况要单独处理
        df_x = dff.iloc[0:1]
        a = (df_x['交易价格'].iloc[:-1] * df_x['剩余数量'].iloc[:-1] ).sum()
        b = df_x['交易价格'].iloc[-1] * (abs(df['交易数量'].iloc[-1]) - df_x['剩余数量'].iloc[:-1].sum())
        c = abs(df['交易数量'].iloc[-1])
        df['成本价格'].iloc[-1] = (a+b) / c
        # 先计算最近一笔的剩余数量，再将之前吃掉的所有单的剩余数量设为0
        d = (abs(df['交易数量'].iloc[-1]) - df_x['剩余数量'].iloc[:-1].sum())
        df['剩余数量'].iloc[0:1][-1:] = df_x['剩余数量'].iloc[-1] - d
        df['剩余数量'].iloc[0:1][:-1] = 0
    return df


def calToday(Date, Data, Type):
    '''
    # 计算当天汇总情况
    :param Date:
    :param Data:
    :param Div:
    :param type:
    :return:
    '''
    # 对Data数据集删除空行，对证券代码标准化
    Data.dropna(inplace=True, subset=['证券代码'])
    Data['证券代码'] = Data['证券代码'].apply(lambda x: nameNorm(x, type=Type))

    Data['成交数量方向'] = ''
    if Type == 'stock':
        Data['产品ID'] = Data['证券代码']

        Data.loc[Data['委托方向'] == '卖出', '成交数量方向'] = -1 * Data.loc[Data['委托方向'] == '卖出', '成交数量']
        Data.loc[Data['委托方向'] == '买入', '成交数量方向'] = Data.loc[Data['委托方向'] == '买入', '成交数量']

    elif Type == 'futures':
        Data.loc[Data['委托方向'] == '卖出开仓', '产品ID'] = Data.loc[Data['委托方向'] == '卖出开仓', '证券代码'] + '.PUT'
        Data.loc[Data['委托方向'] == '卖出平仓', '产品ID'] = Data.loc[Data['委托方向'] == '卖出平仓', '证券代码'] + '.PUT'
        Data.loc[Data['委托方向'] == '买入开仓', '产品ID'] = Data.loc[Data['委托方向'] == '买入开仓', '证券代码'] + '.CALL'
        Data.loc[Data['委托方向'] == '买入平仓', '产品ID'] = Data.loc[Data['委托方向'] == '买入平仓', '证券代码'] + '.CALL'

        Data.loc[Data['委托方向'] == '卖出开仓', '成交数量方向'] = Data.loc[Data['委托方向'] == '卖出开仓', '成交数量']
        Data.loc[Data['委托方向'] == '买入开仓', '成交数量方向'] = Data.loc[Data['委托方向'] == '买入开仓', '成交数量']
        Data.loc[Data['委托方向'] == '卖出平仓', '成交数量方向'] = -1 * Data.loc[Data['委托方向'] == '卖出平仓', '成交数量']
        Data.loc[Data['委托方向'] == '买入平仓', '成交数量方向'] = -1 * Data.loc[Data['委托方向'] == '买入平仓', '成交数量']

    num = pd.DataFrame(Data.groupby(['产品ID'])['成交数量方向'].sum())
    price = pd.DataFrame(Data.groupby(['产品ID'])['成交价格'].sum())
    fee = pd.DataFrame(Data.groupby(['产品ID'])['佣金'].sum()
                       + Data.groupby(['产品ID'])['过户费'].sum()
                       + Data.groupby(['产品ID'])['交割费'].sum()
                       + Data.groupby(['产品ID'])['经手费'].sum()
                       + Data.groupby(['产品ID'])['结算费'].sum()
                       + Data.groupby(['产品ID'])['交易费'].sum()
                       + Data.groupby(['产品ID'])['证管费'].sum()
                       + Data.groupby(['产品ID'])['其他费用'].sum()
                       + Data.groupby(['产品ID'])['全额过户费'].sum())
    tax = pd.DataFrame(Data.groupby(['产品ID'])['印花税'].sum())

    tclose = extractData(Date, Data['证券代码'].drop_duplicates().to_list())
    tfutures_multiplier = extractContractmultiplier(Data['证券代码'].drop_duplicates().to_list())

    if Type == 'futures':
        tclose_ = tclose.copy()
        tclose__ = tclose.copy()
        tclose_['证券代码'] = tclose_['证券代码'] + '.PUT'
        tclose__['证券代码'] = tclose__['证券代码'] + '.CALL'
        tclose = tclose_.append(tclose__)

        tfutures_multiplier_ = tfutures_multiplier.copy()
        tfutures_multipliere__ = tfutures_multiplier.copy()
        tfutures_multiplier_.index = tfutures_multiplier_.index + '.PUT'
        tfutures_multipliere__.index = tfutures_multipliere__.index + '.CALL'
        tfutures_multiplier = tfutures_multiplier_.append(tfutures_multipliere__)

    elif Type == 'stock':
        tfutures_multiplier = pd.DataFrame([1] * len(tclose),index= Data['产品ID'].drop_duplicates())

    close = pd.DataFrame(np.array(tclose['收盘价']), index=tclose['证券代码'])
    today = pd.concat([num, price, fee, tax, close, price, tfutures_multiplier], axis=1)
    today = today.reset_index()
    today.columns = ['证券代码', '交易数量', '交易价格', '交易费用', '印花税', '收盘价', '成本价格', '合约乘数']
    return today


def divCal(Div, Today):
    '''
    考虑现金分红和股票分红
    :param Div:
    :param Today:
    :return:
    '''
    # 对分红数据集进行预处理
    Div['证券代码'] = Div['证券代码'].apply(lambda x: nameNorm(x, type='stock'))

    # 现金分红
    divCash = Div[Div['发生业务'] == '红利到帐']
    divCash.index = divCash['证券代码']
    divCash = divCash['发生金额']
    divCash = divCash.reset_index()
    divCash.columns = ['证券代码', '现金红利']
    TOday = pd.merge(divCash, Today, how='outer')

    # 股票分红
    divStock = data_div[data_div['发生业务'] == '红股上市']
    divStock.index = divStock['证券代码']
    divStock = divStock['发生数量']
    divStock = divStock.reset_index()
    divStock.columns = ['证券代码', '股票股利']
    TODAY = pd.merge(divStock, TOday, how='outer')
    return TODAY


def todayPerform(Date,Position,Today,Type):
    '''

    :param Position:
    :param Today:
    :return:
    '''
    Today['日期'] = Date
    Today.fillna(0,inplace=True)
    asset_not_trade = []

    if Type == 'futures':
        for m in Position['证券代码'].apply(lambda x: x[:10]).drop_duplicates().to_list():
            if m not in Today.loc[Today['交易数量'] != 0]['证券代码'].apply(lambda x: x[:10]).drop_duplicates().to_list():
                asset_not_trade.append(m)
        ydate = Position['日期'].drop_duplicates().to_list()[-1]
        asset_not_trade_ = asset_not_trade.copy()

        if len(asset_not_trade) != 0:
            tclose_not_trade = extractData(Date,asset_not_trade_)
            multiplier = extractContractmultiplier(tclose_not_trade['证券代码'].drop_duplicates().to_list())

            tclose_not_trade_ = tclose_not_trade.copy()
            tclose_not_trade__ = tclose_not_trade.copy()
            tclose_not_trade_['证券代码'] = tclose_not_trade_['证券代码'] + '.PUT'
            tclose_not_trade__['证券代码'] = tclose_not_trade__['证券代码'] + '.CALL'
            tclose_not_trade = tclose_not_trade_.append(tclose_not_trade__)
            tclose_not_trade = pd.DataFrame(np.array(tclose_not_trade['收盘价']), index=tclose_not_trade['证券代码'])

            multiplier_ = multiplier.copy()
            multiplier__ = multiplier.copy()
            multiplier_.index = multiplier_.index + '.PUT'
            multiplier__.index = multiplier__.index + '.CALL'
            multiplier = multiplier_.append(multiplier__)
        else:
            pass

    elif Type == 'stock':
        for m in Position['证券代码'].drop_duplicates().to_list():
            if m not in Today.loc[Today['交易数量'] != 0]['证券代码'].to_list():
                asset_not_trade.append(m)
        ydate = Position['日期'].drop_duplicates().to_list()[-1]

        if len(asset_not_trade) != 0:
            asset_not_trade_ = pd.DataFrame(asset_not_trade).iloc[:, 0].apply(lambda x: x[:10]).to_list()
            close_ = extractData(Date, asset_not_trade)
            tclose_not_trade = pd.DataFrame(np.array(close_['收盘价']), index=close_['证券代码'])
            multiplier = pd.DataFrame([1] * len(tclose_not_trade),index=close_['证券代码'])
        else:
            pass

    if len(asset_not_trade) != 0:
        nanlist = pd.DataFrame([0] * len(tclose_not_trade),index=tclose_not_trade.index)
        asset_not_trade_today = pd.concat([nanlist,nanlist,nanlist,nanlist,tclose_not_trade,nanlist,multiplier],axis=1)
        asset_not_trade_today.reset_index(inplace=True)
        asset_not_trade_today.columns = ['证券代码','交易数量', '交易价格','交易费用','印花税',
                                       '收盘价','成本价格','合约乘数']
        asset_not_trade_today['日期'] = Date

        Today_all = pd.concat([asset_not_trade_today,Today],axis = 0)
    else:
        Today_all = Today.copy()

    Position = pd.concat([Position,Today_all],axis = 0)

    Position.index = list(range(len(Position.index)))
    Position['交易数量'].fillna(0, inplace=True)
    Position['股票股利'].fillna(0, inplace=True)
    Position.loc[:, '持仓数量'] = np.array(pd.DataFrame(Position.groupby(['证券代码'])['交易数量'].cumsum())['交易数量']) + \
                              np.array(pd.DataFrame(Position.groupby(['证券代码'])['股票股利'].fillna(0).cumsum())['股票股利'])
    Position['现金红利'].fillna(0, inplace=True)
    # Position.loc[:,'累计现金红利'] = np.array(pd.DataFrame(Position.groupby(['证券代码'])['现金红利'].cumsum())['现金红利'])

    Position['是否卖出'] = Position['交易数量'] < 0
    Position.loc[Position['交易数量'] == 0, '是否卖出'] = np.nan
    Position_sell = Position.loc[(Position['日期'] == Date) & (Position['是否卖出'] ==True)]['证券代码']
    Position_buy = Position.loc[(Position['日期'] == Date) & (Position['是否卖出'] ==False)]['证券代码']
    for stockj in Position_buy.to_list():
        Position.loc[Position['证券代码'] == stockj, '剩余数量'] = \
        Position.loc[Position['证券代码'] == stockj, '交易数量']
    for stocki in Position_sell.to_list():
        Position.loc[Position['证券代码']== stocki,:] = \
        FIFO(Position.loc[(Position['证券代码']== stocki)]) # & Position['交易数量'] != 0])

    Position['前收盘价'] = Position.groupby(['证券代码'])['收盘价'].shift(1)
    Position.loc[Position['日期'] ==Date,'浮盈'] = Position.loc[Position['日期'] ==Date,:].apply(
        lambda x: x['合约乘数'] * x['持仓数量'] *(x['收盘价']-x['前收盘价']),axis = 1)

    if Type == 'stock':
        Position.loc[Position['日期'] ==Date,'实盈'] = Position.loc[Position['日期'] ==Date,:].apply(
        lambda x:x['现金红利'] + abs(x['交易数量']) * (x['交易价格']-x['成本价格']),axis = 1)


    elif Type == 'futures':
        Position['期权类型'] = Position['证券代码'].apply(lambda x: x.split('.',2)[2])

        Position.loc[(Position['日期'] == Date) & (Position['期权类型'] == 'CALL'), '实盈'] = \
        Position.loc[Position['日期'] == Date, :].apply(
        lambda x: x['交易数量'] * (x['收盘价'] - x['交易价格']), axis=1)

        Position.loc[(Position['日期'] == Date) & (Position['期权类型'] == 'PUT'), '实盈'] = \
        Position.loc[Position['日期'] == Date, :].apply(
        lambda x: - x['交易数量'] * (x['收盘价'] - x['交易价格']), axis=1)
    else:
        raise ValueError('we do not have this type of product!')
    Position.loc[(Position['日期'] == Date)].index = [Date] * len(Position.loc[(Position['日期'] == Date)])
    return Position


def sum(Date, Position, Position_futures):
    '''

    :param Date:
    :param Position:
    :param Position_futures:
    :return:
    '''
    Position.index = Position['日期']
    Position = Position.iloc[:,1:]
    Position_futures.index = Position_futures['日期']
    Position_futures = Position_futures.iloc[:,1:]
    sum_today = Position.loc[Position.index == Date, :].apply(lambda x: x.sum())
    sum_today_futures = Position_futures.loc[Position_futures.index == Date, :].apply(lambda x: x.sum())

    sum = Position.apply(lambda x: x.sum())
    sum_futures = Position_futures.apply(lambda x: x.sum())

    balance = sum_today['实盈'] + sum_today['现金红利'] - sum_today['交易费用'] - sum_today['印花税'] \
              + sum_today_futures['实盈'] + sum_today_futures['现金红利'] - sum_today_futures['交易费用'] - sum_today_futures['印花税']

    all_sum_ = pd.DataFrame({'日期':Date,
                             '账户余额变动': balance,
                             '当日现金红利': sum_today['现金红利'] + sum_today_futures['现金红利'],
                             '当日浮盈': sum_today['浮盈'] + sum_today_futures['浮盈'],
                             '当日实盈': sum_today['实盈'] + sum_today_futures['实盈'],
                             '当日交易佣金': sum_today['交易费用'] + sum_today_futures['交易费用'],
                             '当日印花税': sum_today['印花税'] + sum_today_futures['印花税'],
                             '累计现金红利': sum['现金红利'] + sum_futures['现金红利'],
                             '浮盈': sum_today['浮盈'] + sum_today_futures['浮盈'],
                             '累计实盈': sum['实盈'] + sum_futures['实盈'],
                             '累计交易佣金': sum['交易费用'] + sum_futures['交易费用'],
                             '累计印花税': sum['印花税'] + sum_futures['印花税']},
                              index = [Date])

    return all_sum_


def fixLeg(Begin, End, Principal, Rate):
    '''

    :param Begin:
    :param End:
    :param Principal:
    :param Rate:
    :return:
    '''
    if isinstance(Begin, str) == True:
        begin = datetime.strptime(Begin, "%Y-%m-%d")
    else:
        begin = Begin

    if isinstance(End, str) == True:
        end = datetime.strptime(End, "%Y-%m-%d")
    else:
        end = End
    interval_days = end - begin
    Interest = Principal * ((1 + Rate) ** (interval_days.days / 365) - 1)
    return Interest


if __name__ == '__main__':
    # In[]
    if os.path.exists('./result/all_position.xlsx') == True:
        all_position_ = pd.read_excel('./result/all_position.xlsx')
    else:
        all_position_ = pd.DataFrame(columns=['日期','证券代码','股票股利','现金红利','交易数量',
                                            '交易价格','交易费用','印花税','收盘价','剩余数量',
                                              '成本价格','累计现金红利','浮盈','实盈'])
    if os.path.exists('./result/all_positionfutures.xlsx') == True:
        all_position_futures = pd.read_excel('./result/all_positionfutures.xlsx')
    else:
        all_position_futures = pd.DataFrame(columns=['日期','证券代码','股票股利','现金红利','交易数量',
                                            '交易价格','交易费用','印花税','收盘价','剩余数量',
                                              '成本价格','累计现金红利','浮盈','实盈'])
    all_sum = pd.read_excel('./result/all_sum.xlsx')
    ydate = all_sum.iloc[-1,0]
    # In[]
    if os.path.exists('./data/新综合信息查询_成交回报明细（股票）.xls') == True:
        data_stock = pd.read_excel('./data/新综合信息查询_成交回报明细（股票）.xls')
        tdate_ = data_stock['日期'].drop_duplicates().dropna()  # 取计算当天的日期
        tdate = tdate_.iloc[0]
        today = calToday(Date=tdate, Data=data_stock, Type='stock')
        if os.path.exists('./data/综合信息查询_资金流水20201104.xls') == True:
            data_div = pd.read_excel('./data/综合信息查询_资金流水20201104.xls').dropna(subset=['证券代码'])
            today = divCal(Div=data_div, Today=today)
        else:
            print('ATTENTION: we did not have cash div or stock div today!')
    else:
        today = all_position_.loc[all_position_['日期'] == ydate, '证券代码':'收盘价']
        tdate = input('请输入日期（格式为xxxx-xx-xx）:')
        today['日期'] = [tdate] * len(today)
        tclose = extractData(tdate, today['证券代码'].drop_duplicates().to_list())
        today['收盘价'] = tclose['收盘价']
        print('ATTENTION: we did not trade stocks today!')

    # In[]
    if os.path.exists('./data/新综合信息查询_成交回报明细（股指）.xls') == True:
        data_futures = pd.read_excel('./data/新综合信息查询_成交回报明细（股指）.xls').dropna(subset=['证券代码'])
        tdate_ = data_futures['日期'].drop_duplicates().dropna()  # 取计算当天的日期
        tdate = tdate_.iloc[0]
        today_futures = calToday(Date=tdate, Data=data_futures, Type='futures')
    else:
        today_futures = all_position_futures.loc[all_position_futures['日期'] == ydate, '证券代码':'收盘价']
        tdate = input('请输入日期（格式为xxxx-xx-xx）:')
        today_futures['日期'] = [tdate] * len(today_futures)
        tclose_futures = extractData(tdate, today_futures['证券代码'].drop_duplicates().to_list())
        today_futures['收盘价'] = tclose_futures['收盘价']
        print('ATTENTION: we did not trade futures today!')

    # In[]
    #
    all_position = todayPerform(Date=tdate, Position=all_position_, Today=today, Type='stock')
    all_position_futures = todayPerform(Date=tdate, Position=all_position_futures, Today=today_futures, Type='futures')

    all_position.to_excel('./result/all_position.xlsx',index = False)
    all_position_futures.to_excel('./result/all_positionfutures.xlsx',index = False)

    # # In[]
    all_sum_ = sum(Date=tdate, Position=all_position, Position_futures=all_position_futures)
    all_sum = all_sum.append(all_sum_)
    all_sum.to_excel('./result/all_sum.xlsx',index = False)
    print(all_sum_)

    # In[]
    rate = (313687.579294248 / 383717693.51 + 1) ** (365 / 4) - 1
    fix = fixLeg(Begin='2020-11-01', End=tdate, Principal=383717693.51, Rate=rate)
    print('fix', '%.2f' % fix)
    print('TRS', '%.2f' % (all_sum['账户余额变动'].sum() - fix))

