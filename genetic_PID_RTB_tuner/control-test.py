#!/usr/bin/python
import sys
import random
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

advertiser = "1458"
mode = "test"
ref = 45000
advs_test_bids = 100000
advs_test_clicks = 65
basebid = 69

print "Example of PID control eCPC."
print "Data sample from campaign 1458 from iPinYou dataset."
print "Reference eCPC: " + str(ref)

# parameter setting
minbid = 5
cntr_rounds = 40
para_p = 0.0005
para_i = 0.000001
para_d = 0.0001
div = 1e-6
para_ps = range(0, 40, 5)
para_is = range(0, 25, 5)
para_ds = range(0, 25, 5)
settle_con = 0.1
rise_con = 0.9
min_phi = -2
max_phi = 5


# bidding functions
def lin(pctr, basectr, basebid):
    return int(pctr *  basebid / basectr)


def control_test(cntr_rounds, ref, para_p, para_i , para_d):
    ecpcs = {}
    error_sum = 0.0
    first_round = True
    sec_round = False
    cntr_size = int(len(yp) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    squared_error = 0
    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            phi = 0.0
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            error = ref - ecpcs[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum
            sec_round = False
        else:
            error = ref - ecpcs[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum + para_d*(ecpcs[round-2]-ecpcs[round-1])
        cost = 0
        clks = 0

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        # phi bound
        if phi <= min_phi:
            phi = min_phi
        elif phi >= max_phi:
            phi = max_phi

        for i in range(round*cntr_size, imp_index):
            clk = y[i]
            pctr = yp[i]
            mp = mplist[i]
            bid = max(minbid,lin(pctr, basectr, basebid) * (math.exp(phi)))
            if round == 0:
                bid = 1000.0

            if bid > mp:
                total_wins += 1
                clks += clk
                total_clks += clk
                cost += mp
                total_cost += mp
        ecpcs[round] = total_cost / (total_clks+1)
        error = ecpcs[round] - ref
        squared_error += error * error
    MSE = squared_error / float(cntr_rounds)
    finalECPC = ecpcs[cntr_rounds - 1]
    return MSE, finalECPC


random.seed(10)

mplist = []
y = []
yp = []
mplist_train = []
y_train = []
yp_train = []


#initialize the lr
fi = open("../exp-data/train.txt", 'r')
for line in fi:
    s = line.strip().split()
    y_train.append(int(s[0]))
    mplist_train.append(int(s[1]))
    yp_train.append(float(s[2]))
fi.close()

fi = open("../exp-data/test.txt", 'r')
for line in fi:
    s = line.strip().split()
    y.append(int(s[0]))
    mplist.append(int(s[1]))
    yp.append(float(s[2]))
fi.close()

basectr = sum(yp_train) / float(len(yp_train))

# for reporting
parameters = []
overshoot = []
settling_time = []
rise_time = []
rmse_ss = []
sd_ss = []
report_path = ""


parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t"+str(ref)+"\t" + \
                str(para_p)+"\t"+str(para_i)+"\t"+str(para_d)+"\t"+str(settle_con)+"\t"+str(rise_con)
parameters.append(parameter)
MSE, finalECPC = control_test(cntr_rounds, ref, para_p, para_i, para_d)
print finalECPC
print MSE / 1000000
