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

    # Figure sizes (width, height)
    FIG_SIZE_2_PANEL = (9, 8)
    FIG_SIZE_3_PANEL = (9, 7)

    return fftpack, math, np, plt, FIG_SIZE_2_PANEL, FIG_SIZE_3_PANEL


@app.cell
def _(FIG_SIZE_2_PANEL, FIG_SIZE_3_PANEL, math, np, plt):
    def plot_complex(xs, ys, title='', xl='', yl='', save_name="unknown.svg"):
        fig = plt.figure(figsize=FIG_SIZE_2_PANEL)

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
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=FIG_SIZE_3_PANEL)

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

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=FIG_SIZE_3_PANEL)

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

    def gm_win(xs, ys, lb, gb, sw=10000):
        aq = 1 / sw * len(ys)
        window = []
        for i, y in enumerate(ys):
            t = xs[i] * aq
            fact_1 = -math.pi * t * lb
            fact_2 = (-math.pi * lb * t**2) / (2 * gb * aq)
            window.append(math.exp(fact_1 - fact_2))
        return xs, ys * window

    def sin_win(xs, ys, start=-1.0, end=1.0, power=1):
        num_points = len(ys)
        active_points = int(num_points * end)
        func_points = active_points + active_points * -start
        func_increment = math.pi / func_points
        func_start = func_increment * (func_points - active_points)

        window = np.zeros(num_points)
        for i in range(active_points):
            window[i] = math.sin(func_start + (i * func_increment))**power
        return xs, ys * window

    return (
        damped_sin_wave,
        exp_win,
        gm_win,
        plot_complex,
        plot_real_data_detail_complex,
        plot_real_data_detail_complex_overlay,
        sin_wave,
        sin_win,
        step,
        zero_fill,
    )


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
    mo.vstack([mo.hstack([freq_damped_slider, relax_slider]), fig_damped])
    return


@app.cell
def _(mo):
    mo.md("""
    ## FFT
    """)
    return


@app.cell
def _(mo):
    fft_freq_slider = mo.ui.slider(50, 974, value=512, step=1, label="Frequency")
    return (fft_freq_slider,)


@app.cell
def _(damped_sin_wave, fft_freq_slider, fftpack, plot_complex, zero_fill):
    _relaxation = 10
    _xs_fft, _ysc_fft = damped_sin_wave(fft_freq_slider.value, _relaxation)
    _xs_fft, _ysc_fft = zero_fill(_xs_fft, _ysc_fft, len(_ysc_fft) * 4)
    _yscft = fftpack.fft(_ysc_fft)
    fig_fft = plot_complex(_xs_fft, _yscft, 'FFT of damped cos wave', 'frequency', 'intensity')
    return (fig_fft,)


@app.cell
def _(fft_freq_slider, fig_fft, mo):
    mo.vstack([fft_freq_slider, fig_fft])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Truncation / Step Function
    """)
    return


@app.cell
def _(mo):
    trunc_percent_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Truncation %")
    return (trunc_percent_slider,)


@app.cell
def _(
    damped_sin_wave,
    fftpack,
    np,
    plot_real_data_detail_complex,
    step,
    trunc_percent_slider,
    zero_fill,
):
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
def _(fig_trunc, mo, trunc_percent_slider):
    mo.vstack([trunc_percent_slider, fig_trunc])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exponential Window
    """)
    return


@app.cell
def _(mo):
    exp_lb_slider = mo.ui.slider(0, 20, value=0, step=1, label="Line Broadening (LB)")
    exp_percent_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Truncation %")
    return exp_lb_slider, exp_percent_slider


@app.cell
def _(
    damped_sin_wave,
    exp_lb_slider,
    exp_percent_slider,
    exp_win,
    fftpack,
    np,
    plot_real_data_detail_complex_overlay,
    step,
    zero_fill,
):
    _frequency = 50
    _relaxation = 5
    _xs_exp, _ysc_exp = damped_sin_wave(_frequency, _relaxation)
    _ysc0_exp = np.array(_ysc_exp, copy=True)

    _xs_exp, _ysc0_exp = step(_xs_exp, _ysc0_exp, exp_percent_slider.value)
    _rysc0_exp_pre = np.real(_ysc0_exp)

    _xs_exp, _ysc_exp = step(_xs_exp, _ysc_exp, exp_percent_slider.value)
    _xs_exp, _ysc_exp = exp_win(_xs_exp, _ysc_exp, exp_lb_slider.value)
    _rysc_exp_pre = np.real(_ysc_exp)

    _xs_exp, _ysc_exp = zero_fill(_xs_exp, _ysc_exp, len(_ysc_exp) * 4)
    _yscft_exp = fftpack.fft(_ysc_exp)

    _xs_exp, _ysc0_exp = zero_fill(_xs_exp, _ysc0_exp, len(_ysc0_exp) * 4)
    _ysc0ft_exp = fftpack.fft(_ysc0_exp)

    fig_exp = plot_real_data_detail_complex_overlay(_xs_exp, [_rysc0_exp_pre, _rysc_exp_pre],
                                                     [_ysc0ft_exp, _yscft_exp],
                                                     'Exponential Window', detail=(0.04, 0.06))
    return (fig_exp,)


@app.cell
def _(exp_lb_slider, exp_percent_slider, fig_exp, mo):
    mo.vstack([mo.hstack([exp_lb_slider, exp_percent_slider]), fig_exp])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Gaussian Window (GM)
    """)
    return


@app.cell
def _(mo):
    gm_lb_slider = mo.ui.slider(-40, 20, value=0, step=0.1, label="GM-LB")
    gm_gb_slider = mo.ui.slider(0.0001, 1.0, value=0.1, step=0.01, label="GM-GB")
    gm_percent_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Truncation %")
    return gm_gb_slider, gm_lb_slider, gm_percent_slider


@app.cell
def _(
    damped_sin_wave,
    fftpack,
    gm_gb_slider,
    gm_lb_slider,
    gm_percent_slider,
    gm_win,
    np,
    plot_real_data_detail_complex_overlay,
    step,
    zero_fill,
):
    _frequency = 50
    _relaxation = 5
    _xs_gm, _ysc_gm = damped_sin_wave(_frequency, _relaxation)
    _ysc0_gm = np.array(_ysc_gm, copy=True)

    _xs_gm, _ysc0_gm = step(_xs_gm, _ysc0_gm, gm_percent_slider.value)
    _rysc0_gm_pre = np.real(_ysc0_gm)

    _xs_gm, _ysc_gm = step(_xs_gm, _ysc_gm, gm_percent_slider.value)
    _xs_gm, _ysc_gm = gm_win(_xs_gm, _ysc_gm, gm_lb_slider.value, gm_gb_slider.value)
    _rysc_gm_pre = np.real(_ysc_gm)

    _xs_gm, _ysc_gm = zero_fill(_xs_gm, _ysc_gm, len(_ysc_gm) * 4)
    _yscft_gm = fftpack.fft(_ysc_gm)

    _xs_gm, _ysc0_gm = zero_fill(_xs_gm, _ysc0_gm, len(_ysc0_gm) * 4)
    _ysc0ft_gm = fftpack.fft(_ysc0_gm)

    fig_gm = plot_real_data_detail_complex_overlay(_xs_gm, [_rysc0_gm_pre, _rysc_gm_pre],
                                                    [_ysc0ft_gm, _yscft_gm],
                                                    'Gaussian Window', detail=(0.04, 0.06))
    return (fig_gm,)


@app.cell
def _(fig_gm, gm_gb_slider, gm_lb_slider, gm_percent_slider, mo):
    mo.vstack([mo.hstack([gm_lb_slider, gm_gb_slider, gm_percent_slider]), fig_gm])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sine Window
    """)
    return


@app.cell
def _(mo):
    sin_start_slider = mo.ui.slider(-1.0, 0.0, value=-1.0, step=0.01, label="Start")
    sin_end_slider = mo.ui.slider(0.01, 1.0, value=1.0, step=0.01, label="End")
    sin_power_slider = mo.ui.slider(1, 10, value=1, step=1, label="Power")
    sin_percent_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Truncation %")
    return (
        sin_end_slider,
        sin_percent_slider,
        sin_power_slider,
        sin_start_slider,
    )


@app.cell
def _(
    damped_sin_wave,
    fftpack,
    np,
    plot_real_data_detail_complex_overlay,
    sin_end_slider,
    sin_percent_slider,
    sin_power_slider,
    sin_start_slider,
    sin_win,
    step,
    zero_fill,
):
    _frequency = 50
    _relaxation = 5
    _xs_sin, _ysc_sin = damped_sin_wave(_frequency, _relaxation)
    _ysc0_sin = np.array(_ysc_sin, copy=True)

    _xs_sin, _ysc0_sin = step(_xs_sin, _ysc0_sin, sin_percent_slider.value)
    _rysc0_sin_pre = np.real(_ysc0_sin)

    _xs_sin, _ysc_sin = step(_xs_sin, _ysc_sin, sin_percent_slider.value)
    _xs_sin, _ysc_sin = sin_win(_xs_sin, _ysc_sin, sin_start_slider.value,
                                 sin_end_slider.value, sin_power_slider.value)
    _rysc_sin_pre = np.real(_ysc_sin)

    _xs_sin, _ysc_sin = zero_fill(_xs_sin, _ysc_sin, len(_ysc_sin) * 4)
    _yscft_sin = fftpack.fft(_ysc_sin)

    _xs_sin, _ysc0_sin = zero_fill(_xs_sin, _ysc0_sin, len(_ysc0_sin) * 4)
    _ysc0ft_sin = fftpack.fft(_ysc0_sin)

    fig_sin_win = plot_real_data_detail_complex_overlay(_xs_sin, [_rysc0_sin_pre, _rysc_sin_pre],
                                                         [_ysc0ft_sin, _yscft_sin],
                                                         'Sine Window', detail=(0.04, 0.06))
    return (fig_sin_win,)


@app.cell
def _(
    fig_sin_win,
    mo,
    sin_end_slider,
    sin_percent_slider,
    sin_power_slider,
    sin_start_slider,
):
    mo.vstack([mo.hstack([sin_start_slider, sin_end_slider, sin_power_slider, sin_percent_slider]), fig_sin_win])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Bad First Points
    """)
    return


@app.cell
def _(mo):
    bad_value_slider = mo.ui.slider(-10, 10, value=5.0, step=0.1, label="Bad Value")
    bad_length_slider = mo.ui.slider(0, 100, value=20, step=1, label="Length")
    return bad_length_slider, bad_value_slider


@app.cell
def _(
    bad_length_slider,
    bad_value_slider,
    damped_sin_wave,
    fftpack,
    np,
    plot_real_data_detail_complex_overlay,
    zero_fill,
):
    _frequency = 50
    _relaxation = 5
    _xs_bad, _ysc_bad = damped_sin_wave(_frequency, _relaxation)
    _ysc0_bad = np.array(_ysc_bad, copy=True)

    _rysc0_bad_pre = np.real(_ysc0_bad)

    # Apply bad values to the start
    for i in range(bad_length_slider.value):
        if i < len(_ysc_bad):
            _ysc_bad[i] = bad_value_slider.value

    _rysc_bad_pre = np.real(_ysc_bad)

    _xs_bad, _ysc_bad = zero_fill(_xs_bad, _ysc_bad, len(_ysc_bad) * 4)
    _yscft_bad = fftpack.fft(_ysc_bad)

    _xs_bad, _ysc0_bad = zero_fill(_xs_bad, _ysc0_bad, len(_ysc0_bad) * 4)
    _ysc0ft_bad = fftpack.fft(_ysc0_bad)

    fig_bad = plot_real_data_detail_complex_overlay(_xs_bad, [_rysc0_bad_pre, _rysc_bad_pre],
                                                     [_ysc0ft_bad, _yscft_bad],
                                                     'Bad First Points', detail=(0.04, 0.06))
    return (fig_bad,)


@app.cell
def _(bad_length_slider, bad_value_slider, fig_bad, mo):
    mo.vstack([mo.hstack([bad_value_slider, bad_length_slider]), fig_bad])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Clipping
    """)
    return


@app.cell
def _(mo):
    clip_level_slider = mo.ui.slider(0.01, 1.0, value=1.0, step=0.01, label="Clip Level")
    return (clip_level_slider,)


@app.cell
def _(
    clip_level_slider,
    damped_sin_wave,
    fftpack,
    np,
    plot_real_data_detail_complex_overlay,
    zero_fill,
):
    _frequency = 50
    _relaxation = 5
    _xs_clip, _ysc_clip = damped_sin_wave(_frequency, _relaxation)
    _ysc0_clip = np.array(_ysc_clip, copy=True)

    _rysc0_clip_pre = np.real(_ysc0_clip)

    _ysc_clip = np.clip(_ysc_clip, -clip_level_slider.value, clip_level_slider.value)
    _rysc_clip_pre = np.real(_ysc_clip)

    _xs_clip, _ysc_clip = zero_fill(_xs_clip, _ysc_clip, len(_ysc_clip) * 4)
    _yscft_clip = fftpack.fft(_ysc_clip)

    _xs_clip, _ysc0_clip = zero_fill(_xs_clip, _ysc0_clip, len(_ysc0_clip) * 4)
    _ysc0ft_clip = fftpack.fft(_ysc0_clip)

    _colors_clip = (('0.8', '#1f77b4'), ('0.8', '#1f77b4'), ('0.8', '#1f77b4'))
    fig_clip = plot_real_data_detail_complex_overlay(_xs_clip, [_rysc0_clip_pre, _rysc_clip_pre],
                                                      [_ysc0ft_clip, _yscft_clip],
                                                      'Clipping', detail=(0.04, 0.06),
                                                      colors=_colors_clip)
    return (fig_clip,)


@app.cell
def _(clip_level_slider, fig_clip, mo):
    mo.vstack([clip_level_slider, fig_clip])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Offset Data
    """)
    return


@app.cell
def _(mo):
    offset_slider = mo.ui.slider(0, 50, value=0, step=1, label="Offset")
    return (offset_slider,)


@app.cell
def _(
    damped_sin_wave,
    fftpack,
    np,
    offset_slider,
    plot_real_data_detail_complex_overlay,
    zero_fill,
):
    _frequency = 50
    _relaxation = 10
    _xs_off, _ysc_off = damped_sin_wave(_frequency, _relaxation)
    _ysc0_off = np.array(_ysc_off, copy=True)

    _rysc0_off_pre = np.real(_ysc0_off)

    _ysc_off = np.roll(_ysc_off, offset_slider.value)
    _rysc_off_pre = np.real(_ysc_off)

    _xs_off, _ysc_off = zero_fill(_xs_off, _ysc_off, len(_ysc_off) * 4)
    _yscft_off = fftpack.fft(_ysc_off)

    _xs_off, _ysc0_off = zero_fill(_xs_off, _ysc0_off, len(_ysc0_off) * 4)
    _ysc0ft_off = fftpack.fft(_ysc0_off)

    _colors_off = (('0.8', '#1f77b4'), ('0.8', '#1f77b4'), ('0.8', '#1f77b4'))
    fig_offset = plot_real_data_detail_complex_overlay(_xs_off, [_rysc0_off_pre, _rysc_off_pre],
                                                        [_ysc0ft_off, _yscft_off],
                                                        'Offset Data', detail=(0.04, 0.06),
                                                        colors=_colors_off)
    return (fig_offset,)


@app.cell
def _(fig_offset, mo, offset_slider):
    mo.vstack([offset_slider, fig_offset])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Zero Fills
    """)
    return


@app.cell
def _(mo):
    zero_fills_slider = mo.ui.slider(0, 8, value=0, step=1, label="Zero Fill Level")
    return (zero_fills_slider,)


@app.cell
def _(
    damped_sin_wave,
    fftpack,
    gm_win,
    np,
    plot_real_data_detail_complex_overlay,
    zero_fill,
    zero_fills_slider,
):
    _frequency = 50
    _frequency_offset = 1.25
    _frequency_2 = _frequency + _frequency_offset
    _relaxation = 0.2

    _xs_zf, _ysc_zf = damped_sin_wave(_frequency, _relaxation, data_size=1024*4)
    _xs_zf, _ysc_zf2 = damped_sin_wave(_frequency_2, _relaxation, data_size=1024*4)
    _ysc_zf = _ysc_zf + _ysc_zf2

    _xs_zf, _ysc_zf = gm_win(_xs_zf, _ysc_zf, -0.20, 0.05)
    _ysc0_zf = np.array(_ysc_zf, copy=True)

    _rysc0_zf_pre = np.real(_ysc0_zf)
    _rysc_zf_pre = np.real(_ysc_zf)

    _zero_fill_length = (len(_ysc_zf) * (2**zero_fills_slider.value)) - len(_ysc_zf)

    _xs_zf, _ysc_zf = zero_fill(_xs_zf, _ysc_zf, _zero_fill_length)
    _yscft_zf = fftpack.fft(_ysc_zf)

    _xs_zf, _ysc0_zf = zero_fill(_xs_zf, _ysc0_zf, _zero_fill_length)
    _ysc0ft_zf = fftpack.fft(_ysc0_zf)

    _colors_zf = (('0.8', '#1f77b4'), ('0.8', '#1f77b4'), ('0.8', '#1f77b4'))
    fig_zerofill = plot_real_data_detail_complex_overlay(_xs_zf, [_rysc0_zf_pre, _rysc_zf_pre],
                                                          [_ysc0ft_zf, _yscft_zf],
                                                          'Zero Fills', detail=(0.01, 0.015),
                                                          colors=_colors_zf)
    return (fig_zerofill,)


@app.cell
def _(fig_zerofill, mo, zero_fills_slider):
    mo.vstack([zero_fills_slider, fig_zerofill])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Noise
    """)
    return


@app.cell
def _(mo):
    noise_percent_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Truncation %")
    noise_level_slider = mo.ui.slider(0.0, 3.0, value=0.1, step=0.01, label="Noise Level")
    return noise_level_slider, noise_percent_slider


@app.cell
def _(
    damped_sin_wave,
    fftpack,
    noise_level_slider,
    noise_percent_slider,
    np,
    plot_real_data_detail_complex,
    step,
    zero_fill,
):
    _frequency = 50
    _relaxation = 5
    _xs_noise, _ysc_noise = damped_sin_wave(_frequency, _relaxation)
    _rysc_noise = np.real(_ysc_noise)

    _noise = np.random.normal(size=(len(_rysc_noise),)) * noise_level_slider.value
    _rysc_noise += _noise

    _xs_noise, _ysc_noise = step(_xs_noise, _ysc_noise, noise_percent_slider.value)
    _rysc_noise = np.real(_ysc_noise)
    _xs_noise, _ysc_noise = zero_fill(_xs_noise, _ysc_noise, len(_ysc_noise) * 4)
    _yscft_noise = fftpack.fft(_ysc_noise)

    fig_noise = plot_real_data_detail_complex(_xs_noise, _rysc_noise, _yscft_noise,
                                               'Noise', detail=(0.02, 0.08))
    return (fig_noise,)


@app.cell
def _(fig_noise, mo, noise_level_slider, noise_percent_slider):
    mo.vstack([mo.hstack([noise_percent_slider, noise_level_slider]), fig_noise])
    return


if __name__ == "__main__":
    app.run()
