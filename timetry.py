import time
start = time.clock()

while True:
    elapsed = (time.clock() - start)
    if elapsed > 10 :
        elapsed = 0
        start = time.clock()
    print(elapsed)
    

