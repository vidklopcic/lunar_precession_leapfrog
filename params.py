# zacetni parametri

"""
SOURCE: https://ssd.jpl.nasa.gov/horizons.cgi
"""

# 2458691.500000000 = A.D. 2019-Jul-27 00:00:00.0000 TDB
# X = 5.597390809621808E-01 Y =-8.475032990154722E-01 Z = 3.963355691673358E-05
# VX= 1.408153942006787E-02 VY= 9.413170146019384E-03 VZ=-9.061248384474055E-07
# LT= 5.865975591131264E-03 RG= 1.015662189002483E+00 RR=-9.422902257572297E-05
vz = [24381.57770162345, 16298.497805339213, -1.56891604652909]
xz = [83735774659.51717, -126784688943.94005, 5929095.723010601]

# 2458691.500000000 = A.D. 2019-Jul-27 00:00:00.0000 TDB
# X = 5.611714991394310E-01 Y =-8.453640030262498E-01 Z =-1.452057100706477E-04
# VX= 1.357555850731350E-02 VY= 9.707401741565039E-03 VZ= 3.558660152904129E-05
# LT= 5.860241090645367E-03 RG= 1.014669291022282E+00 RR=-5.795746935580273E-04
vl = [23505.493590941784, 16807.947113050945, 61.616664516133575]
xl = [83950061368.78575, -126464654819.15533, -21722465.040050443]


v_zemlja = []
name = '20 let'
m_s = 1.989e30
m_z = 5.9722e24
m_l = 7.3477e22

period = 100 * 365.24 * 24 * 60 * 60
timestep = 60*60
