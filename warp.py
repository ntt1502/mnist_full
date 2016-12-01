import numpy as np

#diem nhan duoc sau score function
prediction = np.array([0.1, 0.2, 0.12, 0.09, 0.02, 0.3, 0.3, 0.4, 0.16, 0.18])
#cac nhan duoc gan nhan hay khong
target_var = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

true_label = np.argmax(target_var)
predicted_label = np.argmax(test_prediction)

#diem cua nhan dung
f_y = prediction[true_label]

#10 label => Y = 9
Y = 9
N = 0
err = 0
#diem cua cac nhan khong bao gom nhan dung, de pick random vao buoc sau
false_prediction = np.delete(prediction, true_label)

while True:
    #pick random so i
    i = np.random.randint(len(false_prediction))
    #diem cua nhan da duoc pick random
    temp = false_prediction[i]
    N += 1
    if temp + 1 > f_y:
        rank = np.int_(np.floor(Y/N))
        #add trong so L(rank)
        L_rank = 0
        for i in range(rank):
            L_rank += 1/(i+1)
        err += L_rank*(temp + 1 - f_y)
        break
    if N > Y - 1:
        break

print(err)

