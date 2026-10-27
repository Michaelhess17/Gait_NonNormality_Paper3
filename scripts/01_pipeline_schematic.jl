#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "GaitNonNormalityPaper3.jl"))

using .GaitNonNormalityPaper3
using CairoMakie
using LaTeXStrings

const BLUE = RGBf(0.18, 0.43, 0.72)
const RED = RGBf(0.72, 0.16, 0.18)
const GREEN = RGBf(0.18, 0.54, 0.34)
const GOLD = RGBf(0.86, 0.56, 0.12)
const PURPLE = RGBf(0.42, 0.25, 0.62)
const GRAY = RGBf(0.25, 0.25, 0.25)
const LIGHT = RGBf(0.96, 0.97, 0.97)

function panel_label!(ax, label)
    text!(ax, 0.012, 0.982; text=label, fontsize=40, font=:bold,
        color=:black, align=(:left, :top))
end

function schematic_axis(figpos)
    ax = Axis(figpos; limits=(0, 1, 0, 1))
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

function draw_card!(ax, xy, wh; title, subtitle="", color=BLUE)
    x, y = xy
    w, h = wh
    poly!(ax, Rect2f(x, y, w, h);
        color=LIGHT, strokecolor=(color, 0.78), strokewidth=2.8)
    text!(ax, x + 0.055w, y + h - 0.10h;
        text=title, fontsize=32, font=:bold, color=GRAY, align=(:left, :top))
    if !isempty(subtitle)
        text!(ax, x + 0.055w, y + h - 0.24h;
            text=subtitle, fontsize=19, color=(GRAY, 0.82), align=(:left, :top))
    end
    return nothing
end

function draw_arrow!(ax, x1, y1, x2, y2; color=GRAY)
    arrows2d!(ax, Point2f[(x1, y1)], Vec2f[(x2 - x1, y2 - y1)];
        color=(color, 0.72), shaftwidth=8, tipwidth=24, tiplength=28)
    return nothing
end

function mini_measure!(ax, x0, y0, w, h)
    t = range(0, 1; length=140)
    traces = [
        (0.80, 2.0, 0.05, BLUE),
        (0.55, 2.0, 0.38, RED),
        (0.30, 4.0, 0.72, GREEN),
    ]
    for (offset, freq, phase, color) in traces
        y = @. offset + 0.075sin(2pi * (freq * t + phase))
        lines!(ax, x0 .+ w .* t, y0 .+ h .* y; color=color, linewidth=3.0)
    end
    text!(ax, x0, y0 + 0.08h; text="joint angle time series",
        fontsize=15, color=GRAY, align=(:left, :center))
end

function mini_delay_stack!(ax, x0, y0, w, h)
    t = range(0, 1; length=120)
    traces = [
        (0.86, 2.0, 0.05, BLUE),
        (0.56, 2.0, 0.38, RED),
        (0.26, 4.0, 0.72, GREEN),
    ]
    ybase = y0 + 0.050h
    for k in 1:5
        dx = 0.055w * (k - 1)
        dy = -0.095h * (k - 1)
        alpha = 0.22 + 0.13k
        for (offset, freq, phase, color) in traces
            y = @. offset + 0.042sin(2pi * (freq * t + phase + 0.05k))
            lines!(ax, x0 + dx .+ 0.56w .* t, ybase + dy .+ h .* y;
                color=(color, min(alpha, 0.90)), linewidth=k == 5 ? 2.7 : 1.9)
        end
    end
    text!(ax, x0 + 0.50w, y0 + 0.01h; text=L"[y_k,y_{k-1},\ldots,y_{k-d}]",
        fontsize=18, align=(:center, :center), color=GRAY)
end

function mini_operator!(ax, x0, y0, w, h)
    poly!(ax, Rect2f(x0, y0, w, h);
        color=:white, strokecolor=(PURPLE, 0.65), strokewidth=2.0)
    bw, bh = 0.20w, 0.34h
    by = y0 + 0.56h
    boxes = [
        (x0 + 0.04w, L"\mathbf{x}_k", BLUE),
        (x0 + 0.40w, L"A", PURPLE),
        (x0 + 0.76w, L"\mathbf{x}_{k+1}", BLUE),
    ]
    for (bx, txt, color) in boxes
        poly!(ax, Rect2f(bx, by, bw, bh); color=(color, 0.10), strokecolor=color, strokewidth=1.6)
        text!(ax, bx + 0.5bw, by + 0.5bh; text=txt, fontsize=18, font=:bold,
            align=(:center, :center), color=color)
    end
    draw_arrow!(ax, boxes[1][1] + bw + 0.008w, by + 0.5bh, boxes[2][1] - 0.010w, by + 0.5bh)
    draw_arrow!(ax, boxes[2][1] + bw + 0.008w, by + 0.5bh, boxes[3][1] - 0.010w, by + 0.5bh)
    text!(ax, x0 + 0.5w, y0 + 0.36h;
        text=L"\mathbf{x}_{k+1} = A\mathbf{x}_k", fontsize=18, align=(:center, :center), color=GRAY)
    text!(ax, x0 + 0.5w, y0 + 0.15h;
        text=L"\hat{A} = \arg\min_A\|X_{k+1}-AX_k\|_F = X_{k+1}X_k^{+}",
        fontsize=14, align=(:center, :center), color=GRAY)
end

function mini_resolvent!(ax, x0, y0, w, h)
    poly!(ax, Rect2f(x0, y0, w, h);
        color=:white, strokecolor=(GREEN, 0.65), strokewidth=2.0)
    text!(ax, x0 + 0.5w, y0 + 0.62h; text=L"\left\|(zI-A)^{-1}\right\|",
        fontsize=30, align=(:center, :center), color=GREEN)
    text!(ax, x0 + 0.5w, y0 + 0.31h; text="probe response by frequency",
        fontsize=15, align=(:center, :center), color=GRAY)
end

function mini_metrics!(ax, x0, y0, w, h)
    xs = range(0, 1; length=120)
    tg = @. 0.18 + 0.68 * xs * exp(-2.4xs)
    res = @. 0.15 + 0.70 / (1 + 65 * (xs - 0.53)^2)
    lines!(ax, x0 .+ 0.45w .* xs, y0 .+ h .* tg; color=RED, linewidth=3.0)
    lines!(ax, x0 + 0.56w .+ 0.40w .* xs, y0 .+ h .* res; color=GREEN, linewidth=3.0)
    text!(ax, x0 + 0.22w, y0 + 0.08h; text=L"\|A^k\|",
        fontsize=19, color=RED, align=(:center, :center))
    text!(ax, x0 + 0.76w, y0 + 0.08h; text="gain, Q, I/O",
        fontsize=17, color=GREEN, align=(:center, :center))
end

function mini_interpret!(ax, x0, y0, w, h)
    hypotheses = [
        (0.78, GOLD, "the spectrum of a limit-cycle fit is dominated\nby the cycle's neutral modes"),
        (0.46, GREEN, "response geometry (transient growth, tuning,\njoint response structure) carries the dynamics"),
        (0.14, PURPLE, "that geometry changes with speed, after stroke,\nand between individuals"),
    ]
    for (gy, color, txt) in hypotheses
        scatter!(ax, [x0 + 0.05w], [y0 + gy*h]; color=color, markersize=11)
        text!(ax, x0 + 0.10w, y0 + gy*h; text=txt,
            fontsize=14, color=GRAY, align=(:left, :center), lineheight=0.95)
    end
end

function draw_pipeline!(ax)
    w = 0.27
    h = 0.28
    xs = (0.055, 0.365, 0.675)
    ytop = 0.61
    ybot = 0.16

    boxes = [
        ((xs[1], ytop), "Measure", "multivariate gait or model state", BLUE, mini_measure!),
        ((xs[2], ytop), "Delay Embed", "build a Hankel state", BLUE, mini_delay_stack!),
        ((xs[3], ytop), "Fit DMD", "estimate a linear map", PURPLE, mini_operator!),
        ((xs[3], ybot), "Probe Operator", "resolvent response", GREEN, mini_resolvent!),
        ((xs[2], ybot), "Summarize", "transient growth and joint structure", RED, mini_metrics!),
        ((xs[1], ybot), "Interpret", "dynamical readout", GREEN, mini_interpret!),
    ]

    for ((x, y), title, sub, color, mini!) in boxes
        draw_card!(ax, (x, y), (w, h); title=title, subtitle=sub, color=color)
        mini!(ax, x + 0.065w, y + 0.11h, 0.87w, 0.46h)
    end

    ymid_top = ytop + 0.43h
    ymid_bot = ybot + 0.43h
    draw_arrow!(ax, xs[1] + w + 0.012, ymid_top, xs[2] - 0.018, ymid_top)
    draw_arrow!(ax, xs[2] + w + 0.012, ymid_top, xs[3] - 0.018, ymid_top)
    draw_arrow!(ax, xs[3] + 0.50w, ytop - 0.020, xs[3] + 0.50w, ybot + h + 0.020)
    draw_arrow!(ax, xs[3] - 0.018, ymid_bot, xs[2] + w + 0.012, ymid_bot)
    draw_arrow!(ax, xs[2] - 0.018, ymid_bot, xs[1] + w + 0.012, ymid_bot)
end

function make_figure1()
    fig = Figure(size=(1500, 980), fontsize=PAPER_FONT_SIZE)
    axA = schematic_axis(fig[1, 1])
    panel_label!(axA, "A")
    draw_pipeline!(axA)
    Label(fig[1, 1, Top()], "Delay-DMD analysis pipeline";
        fontsize=35, font=:bold, padding=(0, 0, 0, 8))
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    fig = make_figure1()
    savefig("Figure 1.svg", fig)
end
