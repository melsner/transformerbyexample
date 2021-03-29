import sys
import os

def findResultFile(dd):
    for root, dirs, files in os.walk(dd):
        for fi in files:
            if fi == "predictions_dev.txt":
                return "%s/%s" % (root, fi)

if __name__ == "__main__":
    dataset = sys.argv[1]
    run = sys.argv[2]

    languages = {}
    for root, dirs, files in os.walk(dataset):
        for fi in files:
            if fi.endswith(".dev"):
                languages[fi.replace(".dev", "")] = (root + "/" + fi)

    logFile = os.path.basename(os.path.abspath(run)) + ".report"
    with open(logFile, "w") as ofh:
        ofh.write("Logging completed jobs from %s\n" % run)

    for dd in sorted(os.listdir(run + "/fine")):
        if dd in languages:
            print(dd)
            os.system("echo 'Results for %s' >> %s" % (dd, logFile))
            resultFile = findResultFile("%s/fine/%s" % (run, dd))
            if resultFile is None:
                os.system("echo 'No result found\n\n' >> %s" % logFile)
            else:
                langFile = languages[dd]
                cmd = "python script/writeOutput.py %s %s >> %s" % (langFile, resultFile, logFile)
                print(cmd)
                os.system(cmd)
                os.system("echo '\n\n' >> %s" % logFile)
