"""Triton GPU kernels"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from triton.runtime import driver
from tapeforge.contract import (
    PRICE_SCALE,
    SIGN_BUY_INITIATED,
    SIGN_SELL_INITIATED,
    TRADE_DTYPE,
)

DEVICE = driver.active.get_active_torch_device()

@triton.jit
def segment_sums(values_ptr, offsets_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    # Load Segment Boundaries
    pid = tl.program_id(axis = 0)
    start, end = tl.load(offsets_ptr + pid), tl.load(offsets_ptr + pid + 1)

    #load values
    indices = start + tl.arange(0, BLOCK_SIZE)
    values = tl.load(values_ptr + indices, mask=indices < end, other=0)

    sums = tl.sum(values, axis = 0)
    tl.store(output_ptr + pid, sums)

@triton.jit
def aggregate_features(price_ptr, quantity_ptr, buyer_is_maker_ptr, offsets_ptr, total_quantity_ptr, event_count_ptr, signed_quantity_ptr, vwap_ptr, valid_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    start, end = tl.load(offsets_ptr + pid), tl.load(offsets_ptr + pid + 1)
    length = end - start

    # calculate quantity sums
    indices = start + tl.arange(0, BLOCK_SIZE)
    quantities = tl.load(quantity_ptr + indices, mask=indices < end, other = 0)
    total_quantity = tl.sum(quantities, axis = 0)
    tl.store(total_quantity_ptr + pid, total_quantity)

    # calculate quantity sums
    buyer_is_maker = tl.load(buyer_is_maker_ptr + indices, mask = indices < end, other = False)
    signed = tl.where(buyer_is_maker, -1, 1)
    signed_quantity = tl.sum(quantities * signed, axis = 0)

    #calculate aggregate numerator sums
    prices = tl.load(price_ptr + indices, mask = indices < end, other = 0) 
    numerators = prices * quantities
    numerator_sum = tl.sum(numerators, axis = 0)

    #calculate vwap
    has_quantity = total_quantity > 0
    denominator = tl.where(has_quantity, total_quantity, 1).to(tl.float64)
    vwap = numerator_sum.to(tl.float64) / denominator /PRICE_SCALE
    vwap = tl.where(has_quantity, vwap, 0.0)

    # store calculated outputs
    tl.store(event_count_ptr + pid, length)
    tl.store(total_quantity_ptr + pid, total_quantity)
    tl.store(signed_quantity_ptr + pid, signed_quantity)
    tl.store(vwap_ptr + pid, vwap)
    tl.store(valid_ptr + pid, length > 0)

