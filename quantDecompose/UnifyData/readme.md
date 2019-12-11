# UnifyData
This module is trying to transform one DataFrame to one DataFrame with the unified column names and standard period columns.

## Inputs
- **df**: old DataFrame
- **mapping_csv**: a path of csv where one row has the different names for the same name. The first one is the one we want to output.
- **max_period**: the max of periods

## Output: self.data
A DataFrame with the standard column names (the first column in csv) with **max_period** periods. The missing columns are filled by **np.nan**. 

## Mapping CSV
Period numbers are replaced by "#" to represent. The first one in one line is the standard one, the followings in the row are all the possible names.

## Samples
Please see Test/UD_...