
# https://www.geeksforgeeks.org/subarray-whose-absolute-sum-is-closest-to-k/
# Improve with nlogn solution at https://www.geeksforgeeks.org/subarray-whose-sum-is-closest-to-k/
# TODO
def getSubArray(arr, n, K): 
        currSum = 0
        prevDif = 0
        currDif = 0
        result = [-1, -1, abs(K-abs(currSum))] 
        resultTmp = result 
        i = 0
        j = 0
        while(i<= j and j<n): 
            currSum += arr[j] 
            prevDif = currDif 
            currDif = K - abs(currSum) 
            if(currDif <= 0): 
                if abs(currDif) < abs(prevDif): 
                    resultTmp = [i, j, currDif] 
                else: 
                    resultTmp = [i, j-1, prevDif] 
                currSum -= (arr[i]+arr[j])                 
                i += 1
            else: 
                resultTmp = [i, j, currDif] 
                j += 1                
            if(abs(resultTmp[2]) < abs(result[2])): 
                result = resultTmp 
        return result