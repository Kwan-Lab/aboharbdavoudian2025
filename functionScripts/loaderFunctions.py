def loadLightSheetData(**kwargs):
    # A function which loads lightsheet data from specifiied files.
    # Input includes path to files, alongside numbers defining which code is required to preprocess it.

    print('Data being loaded...')


def createDirs(rootDir, switchStruct):

    if switchStruct['testSplit'] or switchStruct['batchSplit'] or switchStruct['scalingFactor'] or switchStruct['oldBatch2']:
        # tsplitTag = dsplitTag = b3tag = scaleTag = ''
        tsplitTag = dsplitTag = scaleTag = batch2Tag = ''

        if switchStruct['testSplit']:
            tsplitTag = 'testSplit'

        if switchStruct['batchSplit']:
            dsplitTag = 'split'

        if switchStruct['scalingFactor']:
            scaleTag = 'scaled'

        if switchStruct['oldBatch2']:
            batch2Tag = 'oldB2'

        # if includeBatch3:
        #     b3tag = 'B3'

        # stringVar = (tsplitTag, dsplitTag, tsplitTag, b3tag)
        stringVar = (scaleTag, dsplitTag, tsplitTag, batch2Tag)
        stringVar = [i for i in stringVar if i]
        dirString = str(len(stringVar)) + '.' + '_'.join(stringVar) + '_'

    else:
        dirString = '0._'

    tempDir = rootDir + dirString + 'Temp//'
    outDir = rootDir + dirString + 'Output//'
    debugDir = rootDir + dirString + 'Debug//'  # Debugging paths and setup
    debug_outPath = debugDir + 'lightSheet_all_ROI.xlsx'

    return [debugDir, tempDir, outDir, debug_outPath]
