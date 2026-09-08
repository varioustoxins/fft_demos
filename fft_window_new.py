import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Core scientific libraries
    import numpy as np
    import scipy.fftpack as fftpack
    import matplotlib.pyplot as plt
    import math

    # Constants
    DEFAULT_GREY = '0.8'
    DEFAULT_BLUE = '#1f77b4'
    DEFAULT_RED = '#d62728'
    PI = math.pi
    DATA_SIZE = 1024
    return math, np, plt


@app.cell
def _(math, np, plt):
    def plot_complex(xs, ys, title='', xl='', yl='', save_name="unknown.svg"):
        fig = plt.figure()

        ax1 = fig.add_axes([0.1, 1.0, 0.8, 0.4])
        ax1.plot(xs, np.real(ys))
        ax1.set_title('%s [real]' % title)
        ax1.set_xlabel(xl)
        ax1.set_ylabel(yl)

        ax2 = fig.add_axes([0.1, 0.40, 0.8, 0.4])
        ax2.plot(xs, np.imag(ys))
        ax2.set_title('%s [imag]' % title)
        ax2.set_ylabel(yl)
        ax2.set_xlabel(xl)

        return fig

    def sin_wave(frequency, data_size=1024):
        xs = [i/data_size for i in np.arange(data_size)]
        ys = [i * frequency for i in xs]
        ysc = np.cos(ys) + 1j * np.sin(ys)
        return xs, ysc

    def exp_win(xs, ys, lb):
        as_ = [np.exp(-lb*x) for x in xs]
        return xs, ys * as_

    def damped_sin_wave(frequency, relaxation, data_size=1024):
        xs, ysc = sin_wave(frequency * 2.0 * math.pi, data_size=data_size)
        xs, ysc = exp_win(xs, ysc, relaxation)
        return xs, ysc

    return plot_complex, sin_wave, exp_win, damped_sin_wave


@app.cell
def _(mo):
    mo.md("""
    ## Sine Wave
    """)
    return


@app.cell
def _(mo):
    frequency_slider = mo.ui.slider(0.0, 512.0, value=100.0, label="Frequency")
    return (frequency_slider,)


@app.cell
def _(frequency_slider, plot_complex, sin_wave):
    _xs, _ysc = sin_wave(frequency_slider.value)
    fig_sine = plot_complex(_xs, _ysc, 'cos wave', 'time', 'intensity')
    return (fig_sine,)


@app.cell
def _(fig_sine, frequency_slider, mo):
    mo.vstack([frequency_slider, fig_sine])
    return


if __name__ == "__main__":
    app.run()

@app.cell
def _(mo):
    mo.md("## Damped Sine Wave")
    return


@app.cell
def _(mo):
    freq_damped_slider = mo.ui.slider(0.0, 512.0, value=100.0, label="Frequency")
    relax_slider = mo.ui.slider(0.0, 20.0, value=1.0, label="Relaxation")
    return freq_damped_slider, relax_slider


@app.cell
def _(freq_damped_slider, relax_slider, damped_sin_wave, plot_complex):
    _xs_d, _ysc_d = damped_sin_wave(freq_damped_slider.value, relax_slider.value)
    fig_damped = plot_complex(_xs_d, _ysc_d, 'damped cos wave', 'time', 'intensity')
    return (fig_damped,)


@app.cell
def _(mo, freq_damped_slider, relax_slider, fig_damped):
    mo.vstack([freq_damped_slider, relax_slider, fig_damped])
    return
