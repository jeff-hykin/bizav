#!/usr/bin/env -S deno run --allow-all
import { FileSystem } from "https://deno.land/x/quickr@0.4.3/main/file_system.js"
import { stats, sum, spread, normalizeZeroToOne } from "https://deno.land/x/good@0.7.7/math.js"
import { zip } from "https://deno.land/x/good@0.7.7/array.js"

const pathToFile = Deno.args[0]

const result = await FileSystem.read(pathToFile)
const studySourceStrings = result.split('finished with value')
const curves = []
for (const eachSourceString of studySourceStrings) {
    const lines = eachSourceString.split('\n')
    let indexOfbest = NaN
    let jsonString = "[\n"
    for (const eachLine of lines) {
        if (eachLine.startsWith("{") && eachLine.endsWith(",},")) {
            jsonString += eachLine.replace(/,\},/,"},") + "\n"
        } else if (eachLine.startsWith("{") && eachLine.endsWith("},")) {
            jsonString += eachLine + "\n"
        }
    }
    // fix the trailing comma of last value
    jsonString = jsonString.replace(/,\n$/, "\n")
    jsonString += "]"

    try {
        curves.push(JSON.parse(jsonString))
    } catch (error) {
        console.error(`Couldn't parse this json string: ${jsonString}`)
    }
}


// 
// save to file
// 
const [ folders, file_name, extension ] = FileSystem.pathPieces(pathToFile)
const outputPath = `${FileSystem.join(...folders)}/${file_name}.curves.json`
await FileSystem.write({
    path: outputPath,
    data: JSON.stringify(curves,0,4),
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