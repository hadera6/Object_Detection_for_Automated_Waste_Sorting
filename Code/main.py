"""Run the whole pipeline: python Code/main.py  (add --retrain to train again)"""
import data
import error_analysis
import evaluate
import train
import visualize

if __name__ == "__main__":
    data.main()                       # 1. clean labels, split, data.yaml
    train.main()                      # 2. train (skipped if weights/best.pt exists)
    evaluate.main()                   # 3. test-set metrics
    results = error_analysis.main()   # 4. predict test set + error_analysis.csv
    visualize.main(results)           # 5. example images from the same predictions