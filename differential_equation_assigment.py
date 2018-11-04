import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from math import ceil as cl

# Function y' = e^(-sin(x)) - y*cos(x) x0 = 0 y0 = 1.0 X(xmax) = 9.3
x0 = 0.0
y0 = 1.0
X = 9.3


def f(x, y):
    return np.e ** (-np.sin(x)) - y * np.cos(x)


def f_ex(x):
    global x0, y0
    return c0() * np.e ** (-np.sin(x)) + x * np.e ** (-np.sin(x))


def c0():
    global x0, y0
    return y0 / (np.e ** (-np.sin(x0))) - x0


def euler_global_error():
    global N, x, y
    amount_of_steps = np.arange(1, N + 1, 1, dtype=int)
    global_errors = np.zeros(N, dtype=float)
    for i in amount_of_steps:
        N = i
        global_errors[i - 1] = np.max(np.abs(euler() - ex_sol()))
    return amount_of_steps, global_errors


def improved_euler_global_error():
    global N, x, y
    amount_of_steps = np.arange(1, N + 1, 1, dtype=int)
    global_errors = np.zeros(N, dtype=float)
    for i in amount_of_steps:
        N = i
        global_errors[i - 1] = np.max(np.abs(up_euler() - ex_sol()))
    return amount_of_steps, global_errors


def runge_global_error():
    global N, x, y
    amount_of_steps = np.arange(1, N + 1, 1, dtype=int)
    global_errors = np.zeros(N, dtype=float)
    for i in amount_of_steps:
        N = i
        global_errors[i - 1] = np.max(np.abs(runge_kutt() - ex_sol()))
    return amount_of_steps, global_errors


def euler():
    global x, N, y
    x = np.zeros(N, dtype=float)
    y = np.zeros(N, dtype=float)
    x[0] = x0
    y[0] = y0
    eps = np.float64(((X - x0) / N))
    for i in range(1, int((X - x0) / eps)):
        x[i] = x[i - 1] + eps
        y[i] = y[i - 1] + eps * f(x[i - 1], y[i - 1])
    return y


def up_euler():
    global x, N, y
    x = np.zeros(N, dtype=float)
    y = np.zeros(N, dtype=float)
    x[0] = x0
    y[0] = y0
    eps = np.float64((X - x0) / N)
    for i in range(1, int((X - x0) / eps)):
        x[i] = x[i - 1] + eps
        # half-function
        hf = f(x[i - 1] + eps / 2, y[i - 1] + eps / 2 * f(x[i - 1], y[i - 1]))
        y[i] = y[i - 1] + eps * hf
    return y


def ex_sol():
    global x
    return np.array(list(map(f_ex, x)))


def runge_kutt():
    global x, N, y
    x = np.zeros(N, dtype=float)
    y = np.zeros(N, dtype=float)
    x[0] = x0
    y[0] = y0
    eps = np.float64((X - x0) / N)
    for i in range(1, N):
        x[i] = x[i - 1] + eps
        k1 = f(x[i - 1], y[i - 1])
        k2 = f(x[i - 1] + eps / 2, y[i - 1] + eps * k1 / 2)
        k3 = f(x[i - 1] + eps / 2, y[i - 1] + eps * k2 / 2)
        k4 = f(x[i - 1] + eps, y[i - 1] + eps * k3)
        y[i] = y[i - 1] + eps / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# global part of program
# (not in the main function to easier control
# all variables and don't try to pass it throw whole code)
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
N = 10
eps = cl((X - x0) / N)
x = np.arange(x0, X, eps, dtype=float)
y = np.zeros(int((X - x0) / eps), dtype=float)
runge = runge_kutt()
el = euler()
up_el = up_euler()
exact = f_ex(x)
# plot all functions and show legend for every function
runge_graph, = plt.plot(x, runge, label='runge')
euler_graph, = plt.plot(x, el, label='euler')
upgrade_euler, = plt.plot(x, up_el, label='improved-euler')
exact_graph, = plt.plot(x, exact, label='exact')
plt.legend(loc=2, ncol=2, mode="None", borderaxespad=0.)
axeps = plt.axes([0.25, 0.1, 0.65, 0.03])
# add Slider for ez GUI
slider_N = Slider(axeps, 'N', 1, 1000, valinit=N)
ax.set_xlim([x0, X])
ax.set_ylim([-20, 20])


def submity(text):
    """
    Update y0 initial value
    :param text:
    :return:
    """
    global y0, y, x0
    y0 = float(text)
    runge_graph.set_ydata(runge_kutt())
    euler_graph.set_ydata(euler())
    upgrade_euler.set_ydata(up_euler())
    exact_graph.set_ydata(ex_sol())
    fig.canvas.draw_idle()


def submitx(text):
    global x0, x, y, y0
    x0 = float(text)
    runge_graph.set_ydata(runge_kutt())
    runge_graph.set_xdata(x)
    euler_graph.set_ydata(euler())
    euler_graph.set_xdata(x)
    upgrade_euler.set_ydata(up_euler())
    upgrade_euler.set_xdata(x)
    exact_graph.set_ydata(ex_sol())
    exact_graph.set_xdata(x)
    fig.canvas.draw_idle()


def __main__():
    """
    Main part of program which consist of all function calls and
    additional things. Also graph errors.
    """
    axcolor = 'lightgoldenrodyellow'
    slider_N.on_changed(update)
    textbox_axs = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    text_box_y = TextBox(textbox_axs, 'y0', initial="y0")
    text_box_y.on_submit(submity)
    textbox_axs = plt.axes([0.025, 0.7, 0.15, 0.15], facecolor=axcolor)
    text_box_x = TextBox(textbox_axs, 'x0', initial="x0")
    text_box_x.on_submit(submitx)
    plt.show()
    # independent graph with errors
    runge = runge_kutt()
    el = euler()
    up_el = up_euler()
    exact = f_ex(x)
    plt.plot(x, np.abs(exact - el), label='euler_error')
    plt.plot(x, np.abs(exact - up_el), label='improved-euler_error')
    plt.plot(x, np.abs(exact - runge), label='runge_error')
    plt.legend(loc=2, ncol=2, mode="None", borderaxespad=0.)
    plt.show()
    # independent graph with global errors
    amount, max_error = euler_global_error()
    plt.plot(amount, max_error, label='euler-global-error')
    amount, max_error = improved_euler_global_error()
    plt.plot(amount, max_error, label='improved-euler-global-error')
    amount, max_error = runge_global_error()
    plt.plot(amount, max_error, label='runge-global-error')
    plt.legend(loc=2, ncol=2, mode="None", borderaxespad=0.)
    plt.show()


def update(val):
    global N, x, y
    N = int(slider_N.val)
    x = np.arange(x0, X - eps, eps, dtype=float)
    runge_graph.set_ydata(runge_kutt())
    runge_graph.set_xdata(x)
    euler_graph.set_ydata(euler())
    euler_graph.set_xdata(x)
    upgrade_euler.set_ydata(up_euler())
    upgrade_euler.set_xdata(x)
    exact_graph.set_ydata(ex_sol())
    exact_graph.set_xdata(x)
    fig.canvas.draw_idle()


if __name__ == '__main__':
    __main__()
