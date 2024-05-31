with open("kernel-positions.raw", 'r') as kernel:
    with open("positions.raw", "r") as positions:
        accurate = 0
        total = 0
        while True:
            k_val = kernel.readline()
            p_val = positions.readline()
            if not k_val or not p_val:
                break
            total += 1
            if int(k_val) == int(p_val):
                accurate += 1
        print(f"{accurate} / {total}")

'''
>>> with open("kernel-positions.raw", 'r') as kernel:
...     with open("positions.raw", "r") as positions:
...             accurate = 0
...             total = 0
...             while True:
...                     k_val = kernel.readline()
...                     p_val = positions.readline()
...                     if not k_val or not p_val:
...                             break
...                     total += 1
...                     if int(k_val) == int(p_val):
...                             accurate += 1
...             print(f"{accurate} / {total}")
... 
2952 / 3105
>>> 2952/3105*100
95.07246376811594
'''