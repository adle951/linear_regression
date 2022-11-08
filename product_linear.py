import gzip
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from collections import defaultdict


def read(filename):
    data = []
    for line in gzip.open(filename):
        line = eval(line)
        d = {}
        d['helpful'] = line['helpful']
        d['reviewText'] = line['reviewText']
        d['rating'] = line['rating']
        d['reviewerID'] = line['reviewerID']
        data.append(d)
    return data


print('Reading data...')
data = read('reviews.json.gz')
print('Done.')


data_train = data[:int(len(data)*0.7)]

# user average
user_helpful = defaultdict(list)
for d in data_train:
    user = d['reviewerID'] 
    user_helpful[user].append(d['helpful'])

user_average = {}
for u in user_helpful:
    user_average[u] = sum([x['nHelpful'] for x in user_helpful[u]]) / sum([x['outOf'] for x in user_helpful[u]])

# total average
average_helpful = sum([d['helpful']['nHelpful'] for d in data_train]) / sum([d['helpful']['outOf'] for d in data_train])
print('average_helpful:', average_helpful)


def feature(d):
    feat = []
    feat.append(len(d['reviewText']))
    return feat


# split
x = [feature(d) for d in data] # feature
y = [d['helpful']['nHelpful'] / d['helpful']['outOf'] for d in data] # label


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

print(list(map(len, [x_train, x_test, y_train, y_test])))


model = LinearRegression().fit(x_train, y_train)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

#　預測
y_pred = model.predict(x_test)
print('predicted response:', y_pred, sep='\n')


# R^2
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# 用 (預測值-答案)^2 加總 
sum_error = 0
for i in range(len(y_test)):
    sum_error += (y_pred[i] - y_test[i]) ** 2
    
print('sum_error=', sum_error)
