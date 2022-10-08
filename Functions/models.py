
"""
MIT License

Copyright (c) 2022 Pawan Goyal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 
Author: Pawan Goyal
"""
import numpy as np

def pendulum(x,t):
    return [
        x[1],
        -np.sin(x[0]) - 0.05*x[1],
        ] 

def FHN_model(x, t):
    a, b, g, I = (0.8, 0.7, 1 / 25, 0.5)
    dx1 = x[0] - (x[0] ** 3) / 3 - x[1] + I
    dx2 = g * (x[0] + a - b * x[1])
    return np.array([dx1, dx2])


def linear_2D(x, t):
    return [-0.1 * x[0] + 2 * x[1], -2 * x[0] - 0.1 * x[1]]



def cubic_2D(x, t):
    return [
        -0.1 * x[0] ** 3 + 2 * x[1] ** 3,
        -2 * x[0] ** 3 - 0.1 * x[1] ** 3,
    ]
