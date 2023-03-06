# FoF matches in the Hubble Source Catalog

Author: *Tamas Budavari budavari@jhu.edu 12/6/2022*

Login to MAST's portal to access the database at <http://mastweb.stsci.edu/hcasjobs/>

Pick the `HSCv3` context, paste a query below and click 'Quick' for immediate execution.

## In M31

```SQL
select top 1000 t.GroupID, t.TaskID, j.JobID, MatchID, Level, BestLevel, NumSources
from hscv3.xrun.Tasks t 
   join hscv3.xrun.Jobs j on j.TaskID=t.TaskID
   join hscv3.xrun.Matches m on m.JobID=j.JobID
where t.GroupID = 1045904 -- M31 
   and t.Stage=2 -- last step
   and j.Progress=100 -- job completed
   and m.Level = 0 -- orig FoF
   and m.NumSources > 28
```

| GroupID | TaskID | JobID | MatchID       | Level | BestLevel | NumSources |
| :------ | :----- | :---- | :------------ | :---- | :-------- | :--------- |
| 1045904 | 35605  | 35605 | 4000714050879 | 0     | 2         | 29         |
| 1045904 | 35605  | 35605 | 4000714053882 | 0     | 1         | 29         |
| 1045904 | 35605  | 35605 | 4000714062587 | 0     | 2         | 29         |
| 1045904 | 35605  | 35605 | 4000714091301 | 0     | 5         | 34         |
| 1045904 | 35605  | 35605 | 4000714093712 | 0     | 3         | 29         |
| 1045904 | 35605  | 35605 | 4000714096940 | 0     | 4         | 30         |
| 1045904 | 35605  | 35605 | 4000714103046 | 0     | 2         | 29         |
| 1045904 | 35605  | 35605 | 4000714162093 | 0     | 1         | 29         |
| 1045904 | 35605  | 35605 | 4000714191349 | 0     | 1         | 29         |

## In SWEEPS

```SQL
select top 1000 t.GroupID, t.TaskID, j.JobID, MatchID, Level, BestLevel, NumSources
from hscv3.xrun.Tasks t 
   join hscv3.xrun.Jobs j on j.TaskID=t.TaskID
   join hscv3.xrun.Matches m on m.JobID=j.JobID
where t.GroupID = 1040910 -- SWEEPS
   and t.Stage = 2 -- last step
   and j.Progress = 100 -- job completed
   and m.Level = 0 
   and m.NumSources > 400
```

| GroupID | TaskID | JobID | MatchID       | Level | BestLevel | NumSources |
| :------ | :----- | :---- | :------------ | :---- | :-------- | :--------- |
| 1040910 | 35604  | 35604 | 4000710334517 | 0     | 7         | 425        |
| 1040910 | 35604  | 35604 | 4000710334723 | 0     | 27        | 458        |
| 1040910 | 35604  | 35604 | 4000710339930 | 0     | 24        | 424        |

## Simply the largest Ones

```SQL
select top 1000 t.GroupID, t.TaskID, j.JobID, MatchID, Level, BestLevel, NumSources
from hscv3.xrun.Tasks t 
   join hscv3.xrun.Jobs j on j.TaskID=t.TaskID
   join hscv3.xrun.Matches m on m.JobID=j.JobID
where t.Stage = 2 -- last step
   and j.Progress = 100 -- job completed
   and m.Level = 0 -- FoF
   and m.NumSources > 5000
```

| GroupID | TaskID | JobID | MatchID       | Level | BestLevel | NumSources |
| :------ | :----- | :---- | :------------ | :---- | :-------- | :--------- |
| 74633   | 35545  | 35545 | 4205889920    | 0     | 70        | 6526       |
| 52501   | 35298  | 35298 | 4556247718    | 0     | 40        | 5739       |
| 1080307 | 35593  | 35593 | 4178981099    | 0     | 119       | 24576      |
| 1080307 | 35593  | 35593 | 4178981787    | 0     | 82        | 6002       |
| 1080307 | 35593  | 35593 | 4178984227    | 0     | 17        | 7046       |
| 1080307 | 35593  | 35593 | 4187029596    | 0     | 127       | 5313       |
| 1080307 | 35593  | 35593 | 4001027716641 | 0     | 16        | 14479      |
| 1080307 | 35593  | 35593 | 4001413573229 | 0     | 40        | 15780      |
| 439774  | 35599  | 35599 | 4540382441    | 0     | 159       | 5136       |

## Pick a match and fetch its sources

Run this by "quick" and save the results in `csv` format. Strictly speaking the `JobID` isn't needed, but including it will make the query faster.

```SQL
select Level, SubID, SourceID, X, Y, Z
from HSCv3.xrun.MatchLinks
where JobID=35604 and MatchID=4000710334517
order by Level, SourceID
-- 850 for both levels
```

850 row(s)

| Level | SubID | SourceID      | X                    | Y                  | Z                  |
| :---- | :---- | :------------ | :------------------- | :----------------- | :----------------- |
| 0     | 0     | 4000710334517 | -0.00381291801127733 | -0.873027769556957 | -0.487655590800151 |
| 0     | 0     | 4000710334521 | -0.00381016654468352 | -0.873027869172163 | -0.487655433969123 |
| 0     | 0     | 4000710334525 | -0.00380688999952366 | -0.873028113194516 | -0.487655022695916 |
| 0     | 0     | 4000710334706 | -0.00382193030882588 | -0.873027610348576 | -0.487655805274346 |
| 0     | 0     | 4000710589039 | ....                 |                    |                    |

To get only the final/best HSC results, include the constraint `Level = BestLevel` e.g., in

```SQL
select Level, SubID, SourceID, X, Y, Z
from HSCv3.xrun.MatchLinks
where JobID=35604 and MatchID=4000710334517
 and Level=BestLevel -- for best results 
 -- and Level=0 -- for original FoF only
order by Level, SourceID -- better order by Level, SubID, SourceID
```

425 row(s)

| Level | SubID | SourceID      | X                    | Y                  | Z                  |
| :---- | :---- | :------------ | :------------------- | :----------------- | :----------------- |
| 7     | 0     | 4000710334517 | -0.00381291801127733 | -0.873027769556957 | -0.487655590800151 |
| 7     | 4     | 4000710334521 | -0.00381016654468352 | -0.873027869172163 | -0.487655433969123 |
| 7     | 2     | 4000710334525 | -0.00380688999952366 | -0.873028113194516 | -0.487655022695916 |
| 7     | 5     | 4000710334706 | -0.00382193030882588 | -0.873027610348576 | -0.487655805274346 |
| 7     | 2     | 4000710589039 | -0.00380881390116012 | -0.873027973773998 | -0.487655257271705 |

Query with Ra and Dec with coordinates:

```SQL
select m.MatchID, m.SubID, m.RA, m.Dec, s.ImageID, l.SourceID, l.X, l.Y, l.Z, s.sigma
from HSCv3.xrun.Matches m
        join HSCv3.xrun.MatchLinks l on l.MatchID=m.MatchID and l.Level=m.Level and l.JobID=m.JobID and l.SubID=m.SubID
        join HSCv3.whl.Sources s on s.SourceID=l.SourceID
where m.JobID=35604 and m.MatchID=4000710334517
        and m.Level=m.BestLevel -- for best results
order by m.MatchID, m.SubID, s.ImageID, s.SourceID
```

Get data by number of sources.
``` SQL
 
select top 1000 t.GroupID, t.TaskID, j.JobID, MatchID, Level, BestLevel, NumSources
from hscv3.xrun.Tasks t
   join hscv3.xrun.Jobs j on j.TaskID=t.TaskID
   join hscv3.xrun.Matches m on m.JobID=j.JobID
where t.Stage = 2 -- last step
   and j.Progress = 100 -- job completed
   and m.Level = 0 -- FoF
   and m.NumSources between 20 and 30
```

``` SQL
select m.MatchID, m.Level, m.SubID, s.ImageID, l.SourceID, l.X, l.Y, l.Z, s.Sigma, s.RA, s.Dec
from HSCv3.xrun.Matches m
        join HSCv3.xrun.MatchLinks l on l.MatchID=m.MatchID and l.Level=m.Level and l.JobID=m.JobID and l.SubID=m.SubID
        join HSCv3.whl.Sources s on s.SourceID=l.SourceID
where m.JobID=33032 and m.MatchID=6000160767697 -- user input
        and m.Level=m.BestLevel -- for best results
order by m.MatchID, m.SubID, s.ImageID, s.SourceID

```

Group of overlapping images for Hubble Deep Fields:

``` SQL
select * from hscv3.whl.Groups
where TargetName like 'HDF%'
```

Group of images with fewer members:

``` sql
select * from hscv3.whl.Groups
where NumImages between 20 and 30
```

Pick a group and get the sources with updated coordinates:

```sql
select top 10 s.ImageID, c.*
from hscv3.xrun.Tasks t
   join hscv3.xrun.Jobs j on j.TaskID=t.TaskID
   join hscv3.xrun.TempCatalog c on c.JobID=j.JobID
   join hscv3.whl.Sources s on s.SourceID=c.SourceID
where t.GroupID = 79757
   and t.Stage = 2 -- final stage
   and j.Progress = 100 -- completed
order by ImageID
```

When you remove the “top 10” to everything, you might run into limitations...

Then you have to use the “Submit” button instead of “Quick” to save a table in your my db. Specify a name in the text field above the query or in the query using the “into” keyword:

```sql
select s.ImageID, c.*, s.RA, s.Dec
into mydb.MyHDF
from hscv3.xrun.Tasks t
   join hscv3.xrun.Jobs j on j.TaskID=t.TaskID
   join hscv3.xrun.TempCatalog c on c.JobID=j.JobID
   join hscv3.whl.Sources s on s.SourceID=c.SourceID
where t.GroupID = 79757
   and t.Stage = 2 -- final stage
   and j.Progress = 100 -- completed
order by ImageID
```

From there you can export the table later using the tools on the MyDB tab.

If you want the original “uncorrected” directions, there is simpler query to get that, something like

```sql
select top 10 s.ImageID, s.SourceID, s.Instrument, s.Catalog, s.X, s.Y, s.Z, s.Sigma, s.Counts
from hscv3.whl.GroupMembers m
   join hscv3.whl.Sources s on s.ImageID = m.MemberID
where m.GroupID = 79757
```