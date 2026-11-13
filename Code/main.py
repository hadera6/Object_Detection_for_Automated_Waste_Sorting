"""Run the whole pipeline: python Code/main.py  (add --retrain to train again)"""
import data
import evaluate
import train

if __name__ == "__main__":
    data.main()       # 1. clean labels, split, data.yaml
    train.main()      # 2. train (skipped if weights/best.pt exists)
    evaluate.main()   # 3. test-set metrics