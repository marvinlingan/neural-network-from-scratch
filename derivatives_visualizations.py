import matplotlib.pyplot as plt
import numpy as np 

def f(x):
    return 2 * x ** 2

# Generate smooth curve
x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y, label='f(x) = 2x²')

colors = ['k', 'g', 'r', 'b', 'c']

def approximate_tangent_line(x, slope):
    return (slope * x) + b

for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    # Calculate slope (derivative) and y-intercept
    slope = (y2 - y1) / (x2 - x1)
    b = y2 - (slope * x2)

    to_plot = [x1 - 0.9, x1 + 0.9]

    # Plot tangent line and point
    plt.scatter(x1, y1, c=colors[i], label=f"x = {x1}")
    plt.plot(to_plot,
             [approximate_tangent_line(point, slope) for point in to_plot],
             c=colors[i])

    # Add slope label
    plt.text(x1, y1 + 2, f"Slope ≈ {slope:.2f}", color=colors[i], fontsize=9)

# Final plot settings
plt.title("Function and Tangent Lines at Different x")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
