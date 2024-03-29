import numpy as np
import matplotlib.pyplot as plt
import cv2

def computeDifference ( bg_img , input_img ) :
    # Tính toán giá trị khác biệt tuyệt đối giữa hai hình ảnh
    difference_three_channel = np.abs( bg_img - input_img )

    # Chuy ển đổi sự khác biệt 3 kênh sang một kênh bằng cách tính trung bình.
    # Điều này giúp giảm dữ liệu khác biệt xuống một giá trị cường độ đơn lẻ.

    '''
    axis=2 trong mảng đa chiều là truy cập vào độ sâu của nó:
    Khi bạn thực hiện một phép tính như np.sum() với axis=2 trên một ảnh màu, 
    bạn đang yêu cầu NumPy tính tổng các giá trị dọc theo chiều sâu (các kênh màu) cho mỗi pixel. 
    Điều này sẽ cộng giá trị từ mỗi kênh màu của mỗi pixel 
    vd: 
    [[[100, 110, 120],   # Pixel 1 có giá trị RGB
    [130, 140, 150]]]  # Pixel 2 có giá trị RGB

    chạy np.sum() với x = 2
    [[330],  # Tổng giá trị của Pixel 1: 100 + 110 + 120
    [420]]  # Tổng giá trị của Pixel 2: 130 + 140 + 150

    '''
    difference_single_channel = np.sum( difference_three_channel , axis =2) / 3.0

    # Chuyển đổi trở lại sang uint8 để phù hợp với xử lý ảnh.
    difference_single_channel = difference_single_channel.astype('uint8')

    return difference_single_channel
def computeBinaryMask ( difference_single_channel , threshold =15) :
    # Áp dụng ngưỡng ( threshold ) để tạo mask nhị phân
    '''Các điểm ảnh có sự chênh lệch màu sắc lớn hơn hoặc bằng 15 được đặt là 255
    (trắng), còn lại là 0 (đen).'''

    difference_binary = np.where( difference_single_channel >= threshold , 255 , 0)

    # Ghép nối mask nhị phân thành 3 kênh để phù hợp với định dạng RGB.
    # chuyển mảng 2 chiều có 1 giá trị thành mảng 3 chiều 
    # mình sẽ tạo 3 mảng giống nhau bằng cách nhân 3
    # axis = -1 là nó ghép mảng theo trục cuối cùng,  tương đương trong mảng 2 chiều là axis = 1 (nếu mảng 3 chiều thì là axis=2)
    binary_binary = np.stack(( difference_binary ,) *3 , axis = -1)

    return binary_binary

def replaceBackGround ( bg_image1 , bg_image2 , ob_image ) :
    # Tính toán sự khác biệt giữa hình ảnh đối tượng và hình nền thứ nhất
    difference_single_channel = computeDifference ( bg_image1 , ob_image )
    # Tính toán mask nhị phân dựa trên sự khác biêt
    binary_mask = computeBinaryMask ( difference_single_channel )
    # Thay thế hình nền:
    # ở vị trí mà mask là đen (0): giữ hình ảnh đối tượng.
    output = np.where( binary_mask == 255 , ob_image , bg_image2 )
    return output


bg_image1 = cv2.imread('images/background1.png')
bg_image2 = cv2.imread('images/background2.png')
ob_image = cv2.imread('images/object.png')

# Resize kích thuớc ảnh đầu vào
bg_image1 = cv2.resize( bg_image1 , (640 , 480))
bg_image2 = cv2.resize( bg_image2 , (640 , 480))
ob_image = cv2.resize( ob_image , (640 , 480))

# Chuyển đổi ảnh màu từ hệ BRG sang hệ RGB
bg_image1 = cv2.cvtColor ( bg_image1 , cv2.COLOR_BGR2RGB )
bg_image2 = cv2.cvtColor ( bg_image2 , cv2.COLOR_BGR2RGB )
ob_image = cv2.cvtColor ( ob_image , cv2.COLOR_BGR2RGB )

# Tiến hành background subtraction
output_image = replaceBackGround(bg_image1, bg_image2, ob_image)

# Hiển thị kết quả
plt.imshow(output_image)
plt.show()
