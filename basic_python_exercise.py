'''
# Viết function thực hiện đánh giá classification model bằng F1-Score.
def evaluate_classification_model(tp, fp, fn):
    # Kiểm tra kiểu dữ liệu của tp, fp, fn
    if not isinstance(tp, int):
        print('tp must be int')
        return
    if not isinstance(fp, int):
        print('fp must be int')
        return
    if not isinstance(fn, int):
        print('fn must be int')
        return
    
    # Kiểm tra giá trị của tp, fp, fn lớn hơn 0
    if tp <= 0 or fp <= 0 or fn <= 0:
        print('tp and fp and fn must be greater than zero')
        return
    
    # Tính Precision, Recall, và F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1_score:.2f}')

# Examples
print(evaluate_classification_model(2, 3, 4))
print(evaluate_classification_model('a', 3, 4))
print(evaluate_classification_model(2, 'a', 4))
print(evaluate_classification_model(2, 3, 'a'))
print(evaluate_classification_model(2, 3, 0))
print(evaluate_classification_model(2.1, 3, 0))

################################################

# Viết function mô phỏng theo 3 activation function.
import math

# Function kiểm tra x có phải là số hay không
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False
    
# Các hàm activation function
def sigmoid(x):
    return 1 / (1 + math.ẽp(-x))

def relu(x):
    return max(0, x)

def elu(x):
    return x if x > 0 else alpha * (math.exp(x) - 1)

def activation_function(x, function_name):
    # Kiểm tra x có phải là số hay không
    if not is_number(x):
        print('x must be a number')
        return
    
    # Kiểm tra function_name có hợp lệ hay không
    valid_functions = ['sigmoid', 'relu', 'elu']
    if function_name not in valid_functions:
        print(f'{function_name} is not supported')
        return
    
    # Convert x sang kiểu float
    x = float(x)

    # Thực hiện activation function tương ứng
    if function_name == 'sigmoid':
        result = sigmoid(x)
    elif function_name == 'relu':
        result = relu(x)
    elif function_name == 'elu':
        result = elu(x)

    # In ra kết quả
    print(f'{function_name}: f({x}) = {result}')


# Example
x = input('Nhập giá trị x: ')
function_name = input('Nhập tên activation function (sigmoid, relu, elu): ') 
activation_function(x, function_name)

############################################################################

# Viết function lựa chọn regression loss function để tính loss.
import random
import math

def calculate_loss(num_samples, loss_name):
    # Kiểm tra num_samples có phải là số nguyên không
    if not num_samples.isnumeric():
        print('number of samples must be an integer number')
        return
    
    # Chuyển đổi num_samples sang kiểu int
    num_samples = int(num_samples)

    # Tạo danh sách các giá trị predict và target
    predicts = [random.uniform(0, 10) for _ in range(num_samples)]
    targets = [random.uniform(0, 10) for _ in range(num_samples)]

    # Tính toán loss
    if loss_name == 'MAE':
        loss = sum(abs(y - y_hat) for y, y_hat in zip(targets, predicts)) / num_samples
    elif loss_name == ' MSE':
        loss = sum((y - y_hat) ** 2 for y, Y_hat in zip(targets, predicts)) / num_samples
    elif loss_name == 'RMSE':
        loss = math.sqrt(sum((y - y_hat) ** 2 for y, y_hat in zip(targets, predicts)) / num_samples)

    # Print kết quả
    print(f'loss name: {loss_name}')
    for i, (predict, target) in enumerate(zip(predicts, targets)):
        print(f'sample-{i}: predict = {predict:.2f}, target = {target:.2f}')
    print(f'loss: {loss:.2f}')
          
# Example
num_samples = input('Nhập số lượng samples: ')
loss_name = input('Nhập tên los function (MAE, MSE, RMSE): ')
calculate_loss(num_samples, loss_name)

##############################################################

# Viết 4 functions để ước lượng các hàm số sau: sin(x), cos(x), sinh(x), cosh(x).
# Hàm tính giai thừa
def factorial(k):
    if k == 0 or k == 1:
        return 1
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result

# Hàm tính sin(x)
def estimate_sin(x, n):
    sin_x = 0
    for i in range(n):
        term = ((-1) ** i) * (x ** (2 * i + 1)) / factorial(2 * i +1)
        sin_x += term
    return sin_x

# Hàm tính cos(x)
def estimate_cos(x, n):
    cos_x = 0
    for i in range(n):
        term = ((-1) ** i) * (x ** (2 * i)) / factorial(2 * i)
        cos_x += term
    return cos_x

# Hàm tính sinh(x)
def estimate_sinh(x, n):
    sinh_x = 0
    for i in range(n):
        term = (x ** (2 * i + 1)) / factorial(2 * i + 1)
        sinh_x += term
    return sinh_x

# Hàm tính cosh(x)
def estimate_cosh(x, n):
    cosh_x = 0
    for i in range(n):
        term = (x ** (2 * i)) / factorial(2 * i)
        cosh_x += term
    return cosh_x

# Chương trình chính
def main():
    import math

    # Người dùng nhập vào giá trị x và n
    x = float(input('Nhập giá trị x (radian): '))
    n = int(input('Nhập số lần lặp n (số nguyên dương > 0): '))

    if n <= 0:
        print('n must be a positive integer greater than 0')
        return
    
    # Tính toán và in ra kết quả
    print(f'estimate_sin(x = {x}, n = {n}) = {estimate_sin(x, n)}')
    print(f'estimate_cos(x = {x}, n = {n}) = {estimate_cos(x, n)}')
    print(f'estimate_sinh(x = {x}, n = {n}) = {estimate_sinh(x, n)}')
    print(f'estimate_cosh(x = {x}, n = {n}) = {estimate_cosh(x, n)}')

# Chạy chương trình chính
main()

####################################################################

# Viết function thực hiện Mean Difference of n-th Root Error.
def md_nre(y, y_hat, n, p):
    """
    Tính toán Mean Difference of nth Root Error (MD_nRE) cho một cặp giá trị y và y_hat.

    Args:
    y (float): Giá trị thực tế.
    y_hat (float): Giá trị dự đoán.
    n (int): Bậc của căn.
    p (int): Bậc của hàm loss.

    Returns:
    float: Kết quả của hàm loss.
    """
    # Tính căn bậc n của y và y_hat
    root_y = y ** (1 / n)
    root_y_hat = y_hat ** (1 / n)

    # Tính giá trị của hàm loss
    loss = abs(root_y - root_y_hat) ** p

    return loss

# Ví dụ sử dụng hàm md_nre với n = 2 và p = 1
y_values = [100, 50, 20, 5.5, 1.0, 0.6]
y_hat_values = [99.5, 49.5, 19.5, 5.0, 0.5, 0.1]

n = 2
p = 1

# Tính toán và in kết quả cho từng cặp y và y_hat
for y, y_hat in zip(y_values, y_hat_values):
    md_nre_value = md_nre(y, y_hat, n, p)
    mae_value = abs(y - y_hat)
    print(f'y: {y}, y_hat: {y_hat}, MAE: {mae_value}, MD_nRE (n={n}, p={p}): {md_nre_value:.3f}')

#################################################################################################

# Câu hỏi trắc nghiệm
# Câu 1: Viết function thực hiện đánh giá classification model bằng F1-Score. Function nhận vào 3 giá trị tp, fp, fn và trả về F1-score.
def calc_f1_score(tp, fp, fn):
    # Tính Precision và Recall
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    # Tính F1-score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

# Kiểm tra function
assert round(calc_f1_score(tp=2, fp=3, fn=5), 2) == 0.33
print(round(calc_f1_score(tp=2, fp=4, fn=5), 2))


# Câu hỏi 2 : Viết function is_number nhận input có thể là string hoặc một số kiểm tra n (một số) có hợp lệ hay không (vd: n=’10’, is_number(n) sẽ trả về True ngược lại là False). Đầu ra của chương trình sau đây là gì?
def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False

# Kiểm tra function
assert is_number(3) == 1.0
assert is_number(' -2a') == 0.0
print(is_number(1))  # Expected: True
print(is_number('n'))  # Expected: False


# Câu hỏi 3 : Đoạn code dưới đây đang thực hiện activation function nào?
x = -2.0
if x <= 0:
    y = 0.0
else:
    y = x
print(y)
# Relu


# Câu hỏi 4 : Viết function thực hiện Sigmoid Function nhận input là x và return kết quả tương ứng trong Sigmoid Function. Đầu ra của chương trình sau đây là gì?
import math

def calc_sig(x):
    return 1 / (1 + math.exp(-x))

# Testing the function
assert round(calc_sig(3), 2) == 0.95
print(round(calc_sig(2), 2))


# Câu hỏi 5 : Viết function thực hiện Elu Function nhận input là x và return kết quả tương ứng trong Elu Function. Đầu ra của chương trình sau đây là gì khi α = 0.01?
import math

def calc_elu(x, alpha=0.01):
    if x <= 0:
        return alpha * (math.exp(x) - 1)
    else:
        return x

# Testing the function
assert round(calc_elu(1)) == 1
print(round(calc_elu(-1), 2))  # Expected: -0.01


# Câu hỏi 6 : Viết function nhận 2 giá trị x, và tên của activation function act_name activation function chỉ có 3 loại (sigmoid, relu, elu), thực hiện tính toán activation function tương ứng với name nhận được trên giá trị của x và trả kết quả. Đầu ra của chương trình sau đây là gì?
import math

def calc_activation_func(x, act_name):
    if act_name == 'sigmoid':
        return 1 / (1 + math.exp(-x))
    elif act_name == 'relu':
        return max(0, x)
    elif act_name == 'elu':
        alpha = 0.01
        if x <= 0:
            return alpha * (math.exp(x) - 1)
        else:
            return x
    else:
        raise ValueError('Unsupported activation function')
    
 # Testing the function
assert calc_activation_func(x=1, act_name='relu') == 1
print(round(calc_activation_func(x=3, act_name='sigmoid'), 2))  # Expected: 0.95   


# Câu hỏi 7 : Viết function tính absolute error = |y−yˆ|. Nhận input là y và yˆ, return về kết quả absolute error tương ứng. Đầu ra của chương trình sau đây là gì?
def calc_ae(y, y_hat):
    return abs(y - y_hat)

# Testing the function
y = 1
y_hat = 6
assert calc_ae(y, y_hat) == 5

y = 2
y_hat = 9
print(calc_ae(y, y_hat))


# Câu hỏi 8 : Viết function tính squared error = (y−y_hat)^2. Nhận input là y và y_hat, return về kết quả squared error tương ứng. Đầu ra của chương trình sau đây là gì?
def calc_se(y, y_hat):
    return (y - y_hat) ** 2

# Testing the function
y = 4
y_hat = 2
assert calc_se(y, y_hat) == 4

print(calc_se(2, 1))


# Câu hỏi 9 : Dựa vào công thức xấp xỉ cos và điều kiện được giới thiệu. Viết function xấp xỉ cos khi nhận x là giá trị muốn tính cos(x) và n là số lần lặp muốn xấp xỉ. Return về kết quả cos(x) với bậc xấp xỉ tương ứng. Đầu ra của chương trình sau đây là gì?
import math

def approx_cos(x, n):
    cos_x = 0
    for i in range(n):
        term = ((-1) ** i) * (x ** (2 * i)) / math.factorial(2 * i)
        cos_x += term
    return cos_x

# Testing the function
assert round(approx_cos(x=1, n=10), 2) == 0.54

# Expected output: -1.0 for cos(3.14) with 10 iterations
print(round(approx_cos(x=3.14, n=10), 2))


# Câu hỏi 10 : Dựa vào công thức xấp xỉ sin và điều kiện được giới thiệu. Viết function xấp xỉ sin khi nhận x là giá trị muốn tính sin(x) và n là số lần lặp muốn xấp xỉ. Return về kết quả sin(x) với bậc xấp xỉ tương ứng. Đầu ra của chương trình sau đây là gì?
import math

def approx_sin(x, n):
    sin_x = 0
    for i in range(n):
        term = ((-1) ** i) * (x ** (2 * i + 1)) / math.factorial(2 * i + 1)
        sin_x += term
    return sin_x

# Testing the function
assert round(approx_sin(x=1, n=10), 4) == 0.8415

# Expected output: 0.0016 for sin(3.14) with 10 iterations
print(round(approx_sin(x=3.14, n=10), 4))


# Câu hỏi 11 : Dựa vào công thức xấp xỉ sinh và điều kiện được giới thiệu. Viết function xấp xỉ sinh khi nhận x là giá trị muốn tính sinh(x) và n là số lần lặp muốn xấp xỉ. Return về kết quả sinh(x) với bậc xấp xỉ tương ứng. Đầu ra của chương trình sau đây là gì?
import math

def approx_sinh(x, n):
    sinh_x = 0
    for i in range(n):
        term = (x ** (2 * i + 1)) / math.factorial(2 * i + 1)
        sinh_x += term
    return sinh_x

# Testing the function
assert round(approx_sinh(x=1, n=10), 2) == 1.18

# Expected output: 11.53 for sinh(3.14) with 10 iterations
print(round(approx_sinh(x=3.14, n=10), 2))
'''

# Câu hỏi 12 : Dựa vào công thức xấp xỉ cosh và điều kiện được giới thiệu. Viết function xấp xỉ cosh khi nhận x là giá trị muốn tính cosh(x) và n là số lần lặp muốn xấp xỉ. Return về kết quả cosh(x) với bậc xấp xỉ tương ứng. Đầu ra của chương trình sau đây là gì?
import math

def approx_cosh(x, n):
    cosh_x = 0
    for i in range(n):
        term = (x ** (2 * i)) / math.factorial(2 * i)
        cosh_x += term
    return cosh_x

# Testing the function
assert round(approx_cosh(x=1, n=10), 2) == 1.54

# Expected output: 11.59 for cosh(3.14) with 10 iterations
print(round(approx_cosh(x=3.14, n=10), 2))
