#[In]

import numpy as np # (1) numpy 모듈 호출
import pandas as pd # (1) pandas 모듈 호출

#%%
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data' # (2) 데이터 URL을 변수 data_url에 넣기
df_data = pd.read_csv(data_url, sep='\s+', header = None) # (3) csv 데이터 로드
df_data.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO' ,'B', 'LSTAT', 'MEDV'] # (4) 데이터의 열 이름 지정
print(df_data.head())


#%%
df_data['weight_0'] = 1 # (2) weight 0 값 추가
df_data = df_data.drop("MEDV", axis=1) # (3) Y 값 제거
df_matrix = df_data.values # (4) 행렬(Matrix) 데이터로 변환하기
weight_vector = np.random.random_sample((14, 1)) # (5) 가중치 w 생성
df_matrix.dot(weight_vector) # (6) 내적 연산 실행 결과 출력
print(df_matrix)
print(weight_vector)

#%%
test_arr1 = np.array([1,2,3,4,5], dtype=int)
print(test_arr1)
print(test_arr1.shape)

#%%열의 개수가 정확히 맞아야 함
test_arr2 = np.array([[1,2,3], [4,5,6]], dtype=int)
print(test_arr2)

#열 갯수가 같지 않으면 오류 발생
#test_arr3 = np.array([[1,2,3,4],[4,5,6]], dtype=int)
#print(test_arr3)

#%%
l1 = [1, 1.4, '3.14', True, False, None]
#for item in l1:
#   print(type(item), end=', ')
test_arr3 = np.array(l1, dtype=float)
print(test_arr3)
print(test_arr3.dtype)

#%% 데이터 형 자동 변환
11 = [1,4,5,"8"]
test_arr4 = np.array(11, dtype=float)
print(test_arr4)
print(test_arr4[3], type(test_arr4[3]))
print(test_arr4.dtype)
print(test_arr4.shape)


#%% 2차원 배열
12 = [[1,2,5,8], [1,2,5,8], [1,2,5,8]]
test_arr5 = np.array(12, dtype=int) #int64, float64
print(test_arr5)
print(test_arr5.dtype)
print(test_arr5.shape)
a = 999999999999999999999999999999
print(a)

print(type(a))

#%% 3차원 배열
tensor_rank3 = [
[[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]],
[[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]],
[[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]],
[[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]]
]
test_arr6 = np.array(tensor_rank3, int)
print(test_arr6)
print(test_arr6.dtype)
print(test_arr6.shape)
print(test_arr6.ndim) # 차원 수 (rank)
print(test_arr6.size) # 전체 원소 수

import sys
print(sys.getsizeof(test_arr6)) #배열이 차지하는 메모리 크기
a = 3
print(a.itemsize) # int32 -> 4byte, int64 -> 8byte

#%% reshape
import numpy as np
x = np.array([[1,2,3,4], [5,6,7,8]] , dtype=int ) # (2,4)
print(x.shape)
# x1 = x.reshape(4,2) # (4,2) # 2*4 = 4*2
# print(x1)
#x2 = x.reshape(2,2,2) # (2,2,2) # 2*4= 2*2*2
#print(x2)
x3 = x.reshape(-1,) # (8,) 1차원
# print(x3)
# x4 = x.reshape(1,-1) # (8,1) 2차원
# print(x4)
#x3.reshape(2,2)
# ValueError: cannot reshape array of size 8 into shape (2,2)

#%% 1차원 배열 -> 2차원 배열
x = np.array(range(8) , dtype=int ) # (8,)
print(x.shape)
x1 = x.reshape(4,2) # (4,2) # 2*4= 4*2
print(x1)
x2 = x.reshape(2,-1) # (2,2,2) # 2*4 = 2*2*2
print(x2)

x3 = x.reshape(-1,1) # (8,1) 2차원

x4 = x3.flatten() # 1차원
print(x4)

#%% numpy slicing
x = np.array(range(16) , dtype=int ).reshape(4,4) # (4,4
print(x)

x1 = x[0:1, :]
# print(x1)
x2 =x[2:4, :]
# print(x2)
x3 =x[0:2, 0:2]
print(x3)

#%%
X = np.array(range(1, 11), dtype=int).reshape(2,5)
# print(x)
x1 = x[1:5]
# print(x1)
x2 = x[:, 1:10]
print(x2)

#%%
x = np.array(range(15), dtype=int.reshape(3,-1))
print(x)

#%%
rangex = list(range(1, 15, 3))
arrayx =np.arange(1, 10, 0.5)
#lx = list(np.arange(15)) #list로 변환
lx = np.arange(15).tolist() #list로 변환
print(rangex, arrayx, sep='\n')
# print(lx)

#%% ones_likem zeros_like, empty_like
x = np.arange(12).reshape(3,4)
xone = np.ones_like(x)
xzero = np.zeros_like(x)
xempty = np.empty_like(x)
print(xone, xzero, xempty, sep='\n')

#%% np.random.uniform
x1 = np.random.uniform(0,1,(3,4)) #(3,4) shape
#print(x1)
x2 = np.random.uniform(1,10,(3,4)) #(3,4) shape
print(x2)

#%%
x = [[1,2], [3,4]]
y = [[5,6], [7,8]]
xsumy = x + y
print(xsumy) # 리스트는 원소끼리 더해지지 않고, 리스트가 합쳐
# [[6,8], [10, 12]]
xa = np.array(x)
ya = np.array(y)
xsumy = xa + ya
print(xsumy) # numpy 배열은 원소끼리 더해짐

#%% concat, vstack, hstack
v1 = np.array([[1,2,3]] , dtype=int) # (1,3)
v2 = np.array([[4,5,6]] , dtype=int) # (1,3)

np_concat0 = np.concatenate([v1, v2], axis=0) # (2,3)
np_concat1 = np.concatenate([v1, v2], axis=1) # (1,6)
print(np_concat0)
print(np_concat1)
print(np_concat1[0].tolist())

#Ax
x = np.arange(1,13).reshape(4,3)
v = np.arange(10,40,10)
print(x)
print(v)
b1 = x + v # 브로드캐스팅
print(b1)



