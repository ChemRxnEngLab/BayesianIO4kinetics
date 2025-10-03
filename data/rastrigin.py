import numpy as np


def rastrigin(X, params):
    A0 = params[0]
    A = params[1:3]
    omega = params[3:5]
    chi = params[5:7]
    val = A0
    val += np.sum((X - chi) ** 2 - A * np.cos(omega * (X - chi)), axis=1)
    return val


def noisy_rastrigin(X, params, noise_std=8):
    return rastrigin(X, params) + np.random.normal(0, noise_std, size=X.shape[0])


params = [50, 10, 10, 0.8 * np.pi, 1.2 * np.pi, 1, 2]


def main():
    import matplotlib.pyplot as plt

    # Define parameters

    # Create a grid of points
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x, y)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    # Compute Rastrigin function values
    Z = rastrigin(X_grid, params).reshape(X1.shape)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax.contourf(X1, X2, Z, cmap="viridis")
    ax2.contourf(
        X1, X2, noisy_rastrigin(X_grid, params).reshape(X1.shape), cmap="viridis"
    )
    # add a 2D plot for constant y=2 below the contour plot
    x_line = np.linspace(-10, 10, 400)
    y_line = 2 * np.ones_like(x_line)

    ax3 = fig.add_subplot(2, 2, (3, 4))
    ax3.plot(
        x_line, rastrigin(np.vstack([x_line, y_line]).T, params), label="Rastrigin"
    )
    ax3.scatter(
        x_line,
        noisy_rastrigin(np.vstack([x_line, y_line]).T, params),
        label="Noisy Rastrigin",
        s=1,
        color="orange",
    )
    ax3.set_title("Rastrigin Function (y=2)")
    ax3.set_xlabel("X1")
    ax3.set_ylabel("Function Value")
    ax3.legend()
    ax.set_title("Rastrigin Function")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    # add a colorbar
    # plt.colorbar(ax.contourf(X1, X2, Z, cmap="viridis"))
    plt.show()


# visualize this function for a given parameter vector over a 2D grid
if __name__ == "__main__":
    main()
