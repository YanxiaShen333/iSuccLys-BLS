#!/usr/bin/env python
# coding: utf-8

import numpy as np

def FM(seq):

    s = "ACDEFGHIKLMNPQRSTVWYX"
    fm_vec = []
    seq_length = len(seq)-1
    for i in s:
        num = seq.count(i)
        if i == 'K':
            num = num - 1 
        freq = num / seq_length

        fm_vec.append(freq)

    return fm_vec



