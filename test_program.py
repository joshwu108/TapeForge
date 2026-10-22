from tapeforge  import contract, generate, reference, windows

trades = generate.generate_trades(seed=7, n_buckets=5, profile="sparse")

offsets = windows.build_offsets(
    trades["timestamp"],
    start_us=0,
    n_buckets=5,
    width_us=contract.DEFAULT_BUCKET_WIDTH_US
)

features = reference.aggregate_reference(trades, offsets)

print(offsets)
print(features.event_count)
print(features.total_quantity / contract.QUANTITY_SCALE)
print(features.vwap)