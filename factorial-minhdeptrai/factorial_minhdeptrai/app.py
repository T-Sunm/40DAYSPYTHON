import streamlit as st

""" Có thể xuất hiện lỗi tại from factorial_minhdeptrai.factorial import fact, khi chạy chương trình tại
máy cá nhân thì ta chỉ cần dùng from factorial import fact. Nhưng khi build package thì nhất định phải
dùng đường dẫn đầy đủ là from factorial_minhdeptrai.factorial import fact """

from factorial_minhdeptrai.factorial import fact

def main():
    st.title("Factorial Calculator")
    number = st.number_input("Enter a number:",min_value=0,max_value=900)

    if st.button ("Calculate") :
        result = fact(number)
        st.write( f"The factorial of {number} is {result}")
        st.balloons()
if __name__ == "__main__":
    main ()