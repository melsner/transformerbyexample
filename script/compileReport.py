import sys
import os
import glob

def findResultFile(dd):
    for root, dirs, files in os.walk(dd):
        for fi in files:
            if fi == "predictions_dev.txt":
                return "%s/%s" % (root, fi)

if __name__ == "__main__":
    dataset = sys.argv[1]
    run = sys.argv[2]
    if len(sys.argv) > 2:
        runType = sys.argv[3]
    else:
        runType = "*"

    testMode = "GOLD-TEST" in dataset

    languages = {}
    for root, dirs, files in os.walk(dataset):
        for fi in files:
            if fi.endswith(".dev"):
                languages[fi.replace(".dev", "")] = (root + "/" + fi)
            if testMode and fi.endswith(".tst"):
                languages[fi.replace(".tst", "")] = (root + "/" + fi)

    logFile = run.replace("/", "-").replace("*", "") + "-" + runType + ".report"
    print("Log file:", logFile)
    with open(logFile, "w") as ofh:
        ofh.write("Logging completed jobs from %s\n" % run)

    for dd in sorted(glob.glob(run + "/" + runType)):
        langName = os.path.basename(dd)
        rs = runType.replace("*", "")
        langName = langName.replace(rs, "")

        if langName in languages:
            print(dd)
            os.system("echo 'Results for %s' >> %s" % (dd, logFile))
            resultFile = findResultFile(dd)
            if resultFile is None:
                os.system("echo 'No result found\n\n' >> %s" % logFile)
            else:
                langFile = languages[langName]
                cmd = "python script/writeOutput.py %s %s >> %s" % (langFile, resultFile, logFile)
                print(cmd)
                os.system(cmd)
                os.system("echo '\n\n' >> %s" % logFile)
