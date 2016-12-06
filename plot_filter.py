
import matplotlib.pyplot as plt

def plot(X):
    n = 16
    plt.figure(figsize=(20, 4))
    for i in range(n):
        for j in range(n):
            # display original
            ax = plt.subplot(n, n, i*n+j + 1)
            plt.imshow(X[:9801, i*n+j].reshape(99, 99))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
