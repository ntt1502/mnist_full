import numpy as np

#diem cua cac 
test_prediction = np.array([0.1, 0.2, 0.12, 0.09, 0.02, 0.3, 0.3])
#cac nhan duoc gan nhan hay khong
target_var = np.array([0, 0, 0, 0, 0, 1, 0])

true_label = np.argmax(target_var)
predicted_label = np.argmax(test_prediction)

f_y = test_prediction[true_label]

Y = 9
N = 0
err = 0
false_prediction = np.delete(test_prediction, true_label)

while True:
    i = np.random.randint(len(false_prediction))
    temp = false_prediction[i]
    N += 1
    if temp + 0.15 > f_y:
        rank = np.int_(np.floor(Y/N))
        L_rank = 0
        for i in range(rank):
            L_rank += 1/(i+1)
        err += L_rank*(temp + 0.15 - f_y)
        break
    if N > Y - 1:
        break

print(err)

