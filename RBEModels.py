############################################################################
#
# RBE Model library, based on set published in Rorvik et al.
# Rørvik et al, Exploration and application of phenomenological RBE models for proton therapy
# Phys Med Biol, 63, 185013, https://doi.org/10.1088/1361-6560/aad9db
#
# Takes input x-ray alpha, x-ray beta, and proton LET; returns proton alpha and beta
# Sorted in alphabetical order
# Default parameters included as default arguments
#
############################################################################
import numpy as np

# 'Safe' calculation of alpha/beta which protects against division by zero
def safeAOverB(alp,bet):
	if (alp<0.0001*bet):
		aOverB=0.0001
	else:
		if bet<0.0001*alp:
			aOverB=10000
		else:
			aOverB=alp/bet
	return aOverB

def carabeAlphaBeta(alp,bet,LET,p0=0.843,p1=0.154,p2=1.09,p3=0.006,p4=2.686):
	aOverB = safeAOverB(alp,bet)

	carabeAlpha = alp*(p0+p1*(p4/aOverB)*LET)
	carabeBeta  = bet*pow(p2+p3*(p4/aOverB)*LET,2)

	return carabeAlpha,carabeBeta

# NB: Had to divide out extra factor of 10 from p1, due to normalisation issue in paper.
def chenAlphaBeta(alp,bet,LET,p0=0.0013,p1=0.0045):
	if LET==0:
		return alp,bet
	chenAlpha = alp*(1+(1-np.exp(-p0*LET*LET))/(p1*LET) )
	chenBeta = bet

	return chenAlpha,chenBeta

def freseAlphaBeta(alp,bet,LET,p0=0.008, p1=0.5):
	if alp<0.0001: alp=0.0001
	freseAlpha = alp*(1 + p0/alp*(LET-p1))
	freseBeta  = bet
	return freseAlpha,freseBeta

def jonesAlphaBeta(alp,bet,LET,p0=30.28,p1=2.696,p2=3.92, p3=0.06, p4=50):
	# Complex model here - p0 is LET turnover
	# p1 and p2 are main value and exponential scaling for alpha_u
	# p3 and p4 are likewise for beta_u

	if alp<0.0001: alp=0.0001
	if bet<0.0001: bet=0.0001
	alphaU = p1*(1-np.exp(-p2*alp))
	betaU =  p3*(1-np.exp(-p4*bet))

	jonesAlpha = alp*(1+(LET-0.22)/p0*(alphaU/alp-1))
	jonesBeta  = bet*(1+(LET-0.22)/p0*(betaU/bet-1))

	return jonesAlpha, jonesBeta

def mairaniAlphaBeta(alp,bet,LET,p0=0.377):
	# This model also includes helium parameterisations, but we only use proton here
	aOverB = safeAOverB(alp,bet)
	mairaniAlpha = alp*(1+p0/aOverB*LET)
	mairaniBeta  = bet

	return mairaniAlpha,mairaniBeta

def mcNamaraAlphaBeta(alp,bet,LET, p0=0.99064,p1=0.35605,p2=1.1012,p3=-0.00387):
	aOverB = safeAOverB(alp,bet)
			
	mcNamaraAlpha = alp*(p0+p1/aOverB*LET)
	mcNamaraBeta  = bet*pow(p2+p3*np.sqrt(aOverB)*LET,2)

	return mcNamaraAlpha,mcNamaraBeta

# Note: p3 is 0.000074, rather than 0.00074 as in Rorvik review. Typo in body of Peeler
# thesis, this value is taken from figure to match curves
def peelerAlphaBeta(alp,bet,LET,p0=0.75, p1=0.00143, p2=1.24, p3=0.000074):
	aOverB = safeAOverB(alp,bet)

	peelerAlpha = alp*(p0+p1/aOverB*pow(LET,3))
	peelerBeta  = bet*pow((p2+p3*aOverB*pow(LET,3)),2)

	return peelerAlpha,peelerBeta

def rorvikUAlphaBeta(alp,bet,LET,p0=0.645):
	aOverB = safeAOverB(alp,bet)
	rorvikAlpha = alp*(1+p0/aOverB*LET)
	rorvikBeta  = bet
	return rorvikAlpha, rorvikBeta

def rorvikWAlphaBeta(alp,bet,LET,p0=0.578,p1=-0.0808,p2=0.00564,p3=-9.92E-5):
	aOverB = safeAOverB(alp,bet)
	rorvikAlpha = alp*(1+(p0*LET+p1*LET*LET+p2*pow(LET,3)+p3*pow(LET,4))/aOverB)
	rorvikBeta  = bet
	return rorvikAlpha, rorvikBeta

def tillyAlphaBeta(alp,bet,LET,p0=0.309,p1=0.550964):
	# Odd model, has fits for alpha/beta of 2 and 10 Gy separately.
	# Between this, we interpolate
	# Calculate range here, but use proper AB for lower calculation
	if alp<2*bet:
		slopeAB=2
	else:
		if alp>10*bet:
			slopeAB=10
		else:
			slopeAB=alp/bet
	slope = p0 + (slopeAB-2.0)/8.0 * (p1-p0)

	aOverB = safeAOverB(alp,bet)
	tillyAlpha = alp*(1+slope*LET/aOverB)
	tillyBeta = bet

	return tillyAlpha, tillyBeta

def unkelbachModel(alp,bet,LET,p0=0.055):
	unkelbachAlpha = alp*(1+p0*LET)
	unkelbachBeta  = bet*pow((1+p0*LET),2)

	return unkelbachAlpha, unkelbachBeta

def wedenbergAlphaBeta(alp,bet,LET,	p0=0.434):
	aOverB = safeAOverB(alp,bet)

	wedenbergAlpha = alp*(1+p0*LET/aOverB)
	wedenbergBeta  = bet

	return wedenbergAlpha, wedenbergBeta

# Caveat here - wilkens calculated for a single alpha/beta value
# Rorvik generalised version used here. 
def wilkensAlphaBeta(alp,bet,LET,p0=0.892,p1=0.179):
	wilkAlp = alp*(p0 + p1*LET)
	wilkBet = bet
	return wilkAlp, wilkBet