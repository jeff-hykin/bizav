#!/usr/bin/env -S deno run --allow-all
import { FileSystem } from "https://deno.land/x/quickr@0.4.3/main/file_system.js"
import { stats, sum, spread, normalizeZeroToOne } from "https://deno.land/x/good@0.7.7/math.js"
import { zip } from "https://deno.land/x/good@0.7.7/array.js"

const pathToFile = Deno.args[0]

const result = await FileSystem.read(pathToFile)
const studySourceStrings = result.split('A new study created in memory with name')
const studyResults = []
for (const eachSourceString of studySourceStrings) {
    const lines = eachSourceString.split('\n')
    let indexOfbest = NaN
    const trials = []
    for (const eachLine of lines) {
        const bestTrialMatch = eachLine.match(/Best is trial (\d+) with value/)
        if (bestTrialMatch) {
            indexOfbest = bestTrialMatch[1]-0 // group1 converted to number
        }
        
        const valueMatch = eachLine.match(/Trial (\d+) finished with value: ([\-+0-9.e]+) and parameters:/)
        const parameterMatch = eachLine.match(/Trial (?:\d+) finished with value: ([\-+0-9.e]+) and parameters: (.+)\. Best is trial/)
        if (valueMatch) {
            const trial = {}
            const [ _, trialIndex, value ] = valueMatch
            trial.trialIndex = trialIndex-0
            trial.score = value-0
            if (isNonRealNumber(trial.score)) {
                console.log(`Score of ${trial.score}  (trial ${trialIndex}) ignored`)
                continue
            }

            if (parameterMatch) {
                const [ _, parametersString ] = parameterMatch
                const probablyValidJson = parametersString.replace(/'/g,'"')
                try {
                    trial.parameters = JSON.parse(probablyValidJson)
                } catch (error) {
                    console.error(`Issue parsing parameters for trial ${trialIndex}\n    parameter part: ${parametersString}\n    whole string: ${eachLine}\n    replacing single quotes with double quotes was insufficient to convert it to valid json`)
                }
            }
            
            trials.push(trial)
        } else {
            console.log(`No trial info found on line: ${eachLine}`)
        }
    }
    
    const bestTrial = trials.filter(each=>each.trialIndex == indexOfbest)[0]
    const scoreStats = stats(trials.map(each=>each.score))

    // 
    // compute score range (linear and log forms)
    // 
    const numberOfBuckets = 30
    const linearScores       = trials.map(each=>each.score)
    const backwardScoreStats = stats(linearScores.map(each=>-each))
    const methods = {
        linearScores,
        dualLogScores: linearScores.map(each=>fullLogFunction(each)),
        forwardLogScores: linearScores.map(each=>Math.log((each-scoreStats.min)+1)),
        backwardLogScores: linearScores.map(each=>-(Math.log(((-each)-backwardScoreStats.min)+1))),
    }
    const inverseScaling = {
        linearScores: each=>each,
        dualLogScores: each=>inverseFullLogFunction(each),
        forwardLogScores: each=>Math.pow(Math.E, each)+scoreStats.min-1,
        backwardLogScores: each=>-((-Math.pow(Math.E,each))-1+backwardScoreStats.min),
    }
    const scoreRangeObject = {}
    for (const [whichMethod, scores] of Object.entries(methods)) {
        const [       _, bucketRanges        ] = createBuckets({ values: scores, numberOfBuckets: numberOfBuckets, })
        const [ buckets, bucketPercentRanges ] = createBuckets({ values: scores, numberOfBuckets: numberOfBuckets, asPercents: true })
        buckets.reverse()
        bucketRanges.reverse()
        bucketPercentRanges.reverse()
        const bucketAverages = zip(buckets, bucketRanges).map(
            ([eachBucket, eachRange]) => 
                stats(
                    eachRange.map(inverseScaling[whichMethod])
                ).average
        )
        const bucketAveragesAsStrings = formatNumberList({ values: bucketAverages, decimalPlaces: 1, withCommas: true })
        const rangesAsStrings = zip(bucketPercentRanges, bucketAveragesAsStrings).map(
                ([[start, end], bucketAverage]) => [
                    `${Math.round(start)}`.padStart(3," "),
                    `${Math.round(end)}`.padStart(3," "),
                    bucketAverage,
                ]
            ).map(
                ([start, end, average])=>`[${start}-${end}%]: ${average}`
            )
        scoreRangeObject[whichMethod] = Object.fromEntries(zip(rangesAsStrings, buckets.map(each=>each.length)))
    }    

    // 
    // save summary
    //
    if (trials.length != 0) {
        studyResults.push({
            bestTrial,
            scores: {
                stats: {count: trials.length, ...scoreStats},
                scoreRange: scoreRangeObject,
            },
            trials,
        })
    }
}


// 
// save to file
// 
const [ folders, file_name, extension ] = FileSystem.pathPieces(pathToFile)
const outputPath = `${FileSystem.join(...folders)}/${file_name}.json`
await FileSystem.write({
    path: outputPath,
    data: JSON.stringify(studyResults,0,4),
})
console.log(`result written to :${outputPath}`)



// 
// helpers
// 
    function isNonRealNumber(value) {
        value = value-0
        return value !== value || value*2 === value
    }
    function formatNumberList({values, decimalPlaces=null, withCommas=false, padWith=" "}) {
        const maxDecimalsAllowedInJavascript = 100
        function numberWithCommas(stringNumber) {
            stringNumber = `${stringNumber}`
            const stringArray = stringNumber.match(/[\s\S]/g)
            const backwardsNumber = stringArray.reverse().join("")
            const backwardsWithCommas = backwardsNumber.replace(/(\d\d\d)(?=\d)/g, "$1,")
            const forwardsWithCommas = backwardsWithCommas.match(/[\s\S]/g).reverse().join("")
            return forwardsWithCommas
        }

        values = values.map(each=>each-0)
        let numberParts = values.map(each=>{
            if (isNonRealNumber(each)) { // NaN or Infinity
                return {
                    intPart: `${each}`,
                    decimalPart: ``,
                }
            } else {
                const decimalPart = `${(each-Math.trunc(each)).toFixed(maxDecimalsAllowedInJavascript)}`.replace(/^-?0\./,"").replace(/0*$/,"")
                return {
                    intPart: `${BigInt(Math.trunc(each))}`,
                    decimalPart,
                }
            }
        })
        
        if (withCommas) {
            numberParts = numberParts.map(({intPart, decimalPart})=>({
                intPart: numberWithCommas(intPart),
                decimalPart,
            }))
        }
        
        const longestDecimal = Math.max(...numberParts.map(({intPart, decimalPart})=>decimalPart.length))
        if (decimalPlaces == null) {
            decimalPlaces = longestDecimal
        }
        let numberStrings
        if (decimalPlaces == 0) {
            numberStrings = numberParts.map(({intPart})=>intPart)
        } else {
            numberParts = numberParts.map(({intPart, decimalPart})=>({intPart, decimalPart: decimalPart.padEnd(longestDecimal,"0").slice(0,decimalPlaces) }))
            numberStrings = numberParts.map(({intPart, decimalPart})=>{
                if (intPart.match(/NaN|Inf/)) {
                    return intPart
                } else {
                    return `${intPart}.${decimalPart}`
                }
            })
        }

        const maxStringLength = Math.max(...numberStrings.map(each=>each.length))
        return numberStrings.map(each=>each.padStart(maxStringLength, padWith))
    }

    function fullLogFunction(value, base=null) {
        if (value == 0) {
            return 0
        } else if (value > 0) {
            value = value+1
            return !base ? Math.log(value) : Math.log(value)/Math.log(base)
        } else {
            // flip the value
            value = -(value-1)
            const logResult = !base ? Math.log(value) : Math.log(value)/Math.log(base)
            return -logResult
        }
    }

    function inverseFullLogFunction(each, base=Math.E) {
        if (each == 0) {
            return 0
        } else if (each > 0) {
            return Math.pow(base, each) - 1
        } else if (each < 0) {
            return -(Math.pow(base, -each) - 1)
        }
    }
    
    function createBuckets({values, numberOfBuckets, asPercents=false}) {
        var maxValue = Math.max(...values)
        var minValue = Math.min(...values)
        var valueRange = maxValue - minValue
        var bucketSize = valueRange / numberOfBuckets
        const buckets = [...Array(numberOfBuckets)].map(each=>([]))
        if (bucketSize == 0) { // edgecase
            // all the buckets go in the middle
            buckets[ Math.floor(numberOfBuckets/2) ] = values
        } else {
            for (let eachValue of values) {
                const whichBucket = Math.min(
                    Math.floor((eachValue-minValue) / bucketSize),
                    numberOfBuckets-1,
                )
                buckets[whichBucket].push(eachValue)
            }
        }
        let bucketRanges
        if (asPercents) {
            bucketRanges = buckets.map(
                (each, bucketIndex)=>([
                    ((bucketIndex+0)/numberOfBuckets)*100,
                    ((bucketIndex+1)/numberOfBuckets)*100,
                ])
            )
        } else {
            bucketRanges = buckets.map(
                (each,bucketIndex)=>([
                    minValue + (bucketIndex+0)*bucketSize,
                    minValue + (bucketIndex+1)*bucketSize,
                ])
            )
        }
        
        return [ buckets, bucketRanges ]
    }