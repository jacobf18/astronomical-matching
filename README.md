# Astronomical Matching

Collection of algorithms to create matchings between astronomical catalogs.

## Background

Astronomical photos inherintly have error in them. To best predict where objects are, multiple photos of the same part of the sky are combined. This is equivalent to matching objects in one image to objects in another image. Existing methods rely on nearest-neighbor heuristics which do not take into account the fact that two objects from one image cannot be assigned to the same object.


## Documentation

``` sql
-- ACS pixel size * max(2,FWHM_IMAGE) -- Rick's formula
Sigma = CONVERT(float,0.1) * 0.05
                * case when COALESCE(c.FWHM_IMAGE,0) < 2 then 2 else c.FWHM_IMAGE end,
```

ACS pixel size is 0.05 arcsec/pixel. FWHM_IMAGE is in arcsec.

`sigma = 0.05 * max(2, FWHM_IMAGE)`

``` sql
create function xrun.fWeightExpr(@SigmaArcsecExpr varchar(512))
returns varchar(512)
as
begin
        return '2.3*POWER(PI()*' + @SigmaArcsecExpr + '/180/3600,-2)';
end
go
```