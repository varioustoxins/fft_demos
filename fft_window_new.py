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

    def zero_fill(xs, ysc, extra=None):
        if extra is None:
            len_ysc = len(ysc)
            extra = len_ysc // 2
        ysc = np.pad(ysc, extra // 2)
        ysc = np.roll(ysc, -extra // 2)
        new_xs = np.arange(0.0, 1.0, 1.0 / len(ysc))
        return new_xs, ysc

    def step(xs, ys, percentage=0.5):
        len_ys = len(ys)
        step_arr = np.zeros(len_ys)
        for i in range(int(len(ys) * percentage)):
            step_arr[i] = 1.0
        return xs, ys * step_arr

    def plot_real_data_detail_complex(xs, rys, ys, title='', detail=(0.0, 1.0)):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 8))

        rxs = range(len(rys))
        ax0.plot(rxs, rys)
        ax0.set_title(f"{title} (Time Domain Real)")

        ax1.plot(xs, np.real(ys))
        ax1.set_title("Full Spectrum")

        start = int(len(xs) * detail[0])
        end = int(len(xs) * detail[1])
        ax2.plot(xs[start:end], np.real(ys)[start:end])
        ax2.set_title("Detail Spectrum")

        plt.tight_layout()
        return fig

    def plot_real_data_detail_complex_overlay(xs, rys, ys, title='', detail=(0.0, 1.0), colors=None):
        if colors is None:
            colors = (('#d62728', '#1f77b4'), ('0.8', '#1f77b4'), ('0.8', '#1f77b4'))

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 8))

        rxs = range(len(rys[0]))
        ax0.plot(rxs, rys[0], colors[0][0], alpha=0.5)
        ax0.plot(rxs, rys[1], colors[0][1])
        ax0.set_title(f"{title} (Time Domain Real)")

        ax1.plot(xs, np.real(ys[0]), colors[1][0])
        ax1.plot(xs, np.real(ys[1]), colors[1][1])
        ax1.set_title("Full Spectrum")

        start = int(len(xs) * detail[0])
        end = int(len(xs) * detail[1])
        ax2.plot(xs[start:end], np.real(ys[0])[start:end], colors[2][0])
        ax2.plot(xs[start:end], np.real(ys[1])[start:end], colors[2][1])
        ax2.set_title("Detail Spectrum")

        plt.tight_layout()
        return fig

    return (damped_sin_wave, exp_win, plot_complex, plot_real_data_detail_complex,
            plot_real_data_detail_complex_overlay, sin_wave, step, zero_fill)


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


@app.cell
def _(mo):
    mo.md("""
    ## Damped Sine Wave
    """)
    return


@app.cell
def _(mo):
    freq_damped_slider = mo.ui.slider(0.0, 512.0, value=100.0, label="Frequency")
    relax_slider = mo.ui.slider(0.0, 20.0, value=1.0, label="Relaxation")
    return freq_damped_slider, relax_slider


@app.cell
def _(damped_sin_wave, freq_damped_slider, plot_complex, relax_slider):
    _xs_d, _ysc_d = damped_sin_wave(freq_damped_slider.value, relax_slider.value)
    fig_damped = plot_complex(_xs_d, _ysc_d, 'damped cos wave', 'time', 'intensity')
    return (fig_damped,)


@app.cell
def _(fig_damped, freq_damped_slider, mo, relax_slider):
    mo.vstack([freq_damped_slider, relax_slider, fig_damped])
    return


@app.cell
def _(mo):
    mo.md("## FFT")
    return


@app.cell
def _(mo):
    fft_freq_slider = mo.ui.slider(50, 974, value=512, step=1, label="Frequency")
    return (fft_freq_slider,)


@app.cell
def _(fft_freq_slider, damped_sin_wave, zero_fill, fftpack, plot_complex, np):
    _relaxation = 10
    _xs_fft, _ysc_fft = damped_sin_wave(fft_freq_slider.value, _relaxation)
    _xs_fft, _ysc_fft = zero_fill(_xs_fft, _ysc_fft, len(_ysc_fft) * 4)
    _yscft = fftpack.fft(_ysc_fft)
    fig_fft = plot_complex(_xs_fft, _yscft, 'FFT of damped cos wave', 'frequency', 'intensity')
    return (fig_fft,)


@app.cell
def _(mo, fft_freq_slider, fig_fft):
    mo.vstack([fft_freq_slider, fig_fft])
    return


@app.cell
def _(mo):
    mo.md("## Truncation / Step Function")
    return


@app.cell
def _(mo):
    trunc_percent_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Truncation %")
    return (trunc_percent_slider,)


@app.cell
def _(trunc_percent_slider, damped_sin_wave, step, zero_fill, fftpack,
      plot_real_data_detail_complex, np):
    _frequency = 50
    _relaxation = 5
    _xs_tr, _ysc_tr = damped_sin_wave(_frequency, _relaxation)
    _rysc_tr = np.real(_ysc_tr)
    _xs_tr, _ysc_tr = step(_xs_tr, _ysc_tr, trunc_percent_slider.value)
    _rysc_tr = np.real(_ysc_tr)
    _xs_tr, _ysc_tr = zero_fill(_xs_tr, _ysc_tr, len(_ysc_tr) * 4)
    _yscft_tr = fftpack.fft(_ysc_tr)
    fig_trunc = plot_real_data_detail_complex(_xs_tr, _rysc_tr, _yscft_tr,
                                               'Truncated FID', detail=(0.02, 0.08))
    return (fig_trunc,)


@app.cell
def _(mo, trunc_percent_slider, fig_trunc):
    mo.vstack([trunc_percent_slider, fig_trunc])
    return


@app.cell
def _(mo):
    mo.md("## Exponential Window")
    return


@app.cell
def _(mo):
    exp_lb_slider = mo.ui.slider(0, 20, value=0, step=1, label="Line Broadening (LB)")
    exp_percent_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Truncation %")
    return exp_lb_slider, exp_percent_slider


@app.cell
def _(exp_lb_slider, exp_percent_slider, damped_sin_wave, step, exp_win,
      zero_fill, fftpack, plot_real_data_detail_complex_overlay, np):
    _frequency = 50
    _relaxation = 5
    _xs_exp, _ysc_exp = damped_sin_wave(_frequency, _relaxation)
    _ysc0_exp = np.array(_ysc_exp, copy=True)

    _xs_exp, _ysc0_exp = step(_xs_exp, _ysc0_exp, exp_percent_slider.value)
    _xs_exp, _ysc_exp = step(_xs_exp, _ysc_exp, exp_percent_slider.value)
    _xs_exp, _ysc_exp = exp_win(_xs_exp, _ysc_exp, exp_lb_slider.value)
    _rysc_exp = np.real(_ysc_exp)

    _xs_exp, _ysc_exp = zero_fill(_xs_exp, _ysc_exp, len(_ysc_exp) * 4)
    _yscft_exp = fftpack.fft(_ysc_exp)

    _xs_exp, _ysc0_exp = zero_fill(_xs_exp, _ysc0_exp, len(_ysc0_exp) * 4)
    _ysc0ft_exp = fftpack.fft(_ysc0_exp)

    _rysc0_exp = np.real(_ysc0_exp)
    fig_exp = plot_real_data_detail_complex_overlay(_xs_exp, [_rysc0_exp, _rysc_exp],
                                                     [_ysc0ft_exp, _yscft_exp],
                                                     'Exponential Window', detail=(0.04, 0.06))
    return (fig_exp,)


@app.cell
def _(mo, exp_lb_slider, exp_percent_slider, fig_exp):
    mo.vstack([exp_lb_slider, exp_percent_slider, fig_exp])
    return


if __name__ == "__main__":
    app.run()
