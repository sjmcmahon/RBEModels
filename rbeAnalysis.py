import numpy as np 
import scipy.special

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

import RBEModels

sns.set(style="white")

# Three groups - a/b dependence, dose scaling, other.
names = ["Carabe","Mairani","McNamara","RorvikU","RorvikW","Wedenberg",
		 "Tilly","Unkelbach",
		 "Chen","Frese","Jones","Wilkens", "Peeler"]

modelList = [RBEModels.carabeAlphaBeta, RBEModels.mairaniAlphaBeta, RBEModels.mcNamaraAlphaBeta, RBEModels.rorvikUAlphaBeta, RBEModels.rorvikWAlphaBeta, RBEModels.wedenbergAlphaBeta, 
			 RBEModels.tillyAlphaBeta,  RBEModels.unkelbachModel,
			 RBEModels.chenAlphaBeta,   RBEModels.freseAlphaBeta,   RBEModels.jonesAlphaBeta, RBEModels.wilkensAlphaBeta,
	         RBEModels.peelerAlphaBeta ]

# Normalisation point for calculations
normLET = 10
normAB = 5
correlationType = "pearson" # spearman or pearson

# Set random seed to allow reproducibility across runs
np.random.seed(42)

# Couple of methods to help with drawing
def hideCurrentAxis(*args, **kwds):
    plt.gca().set_visible(False)

def colorForText(backgroundColour):
	brightness = 0.299*backgroundColour[0]+0.587*backgroundColour[1]+0.114*backgroundColour[2]
	if brightness<0.65:
		return [1,1,1,1]
	return [0,0,0,1]

# Calculate Mean Inactivation Dose (MID). Include a few small checks in case of unphysical values.
def fullMIDCalc(alpha,beta):
	if beta<1E-6 or abs(alpha/(2*np.sqrt(beta)))>10:
		return 1.0/alpha
	if beta<0:
		beta=-beta
	constant    = np.sqrt(np.pi)/(2*np.sqrt(beta))
	logExponent = (alpha*alpha/(4*beta))
	erf         = scipy.special.erfc(alpha/(2*np.sqrt(beta)))

	if erf==0: return 0

	erfExpProduct = np.exp(np.log(erf)+logExponent)
	return constant*erfExpProduct

# Calculate RBEs for an arbitrary model passed as an argument
def modelRBEs(model,conds,params):
	returnList = []
	for cond in conds:
		alp,bet, L = cond		
		baseMID = fullMIDCalc(alp,bet)
		ap,bp= model(alp,bet,L,*params)
		returnList.append(baseMID/fullMIDCalc(ap,bp))
	return np.array(returnList)

def calculateRBE(model, alp,bet,LET, normalise, normMethod = "LET"):
	baseMID = fullMIDCalc(alp,bet)

	ap,bp= model(alp,bet,LET)

	protonMID = fullMIDCalc(ap,bp)
	RBE = baseMID/protonMID

	if normalise:
		if normMethod=="LET":
			# Normalise to focus on LET
			apn,bpn= model(alp,bet,normLET)
			refMID =  fullMIDCalc(apn,bpn)
			refRBE = baseMID/refMID
			RBE = (RBE-1)/(refRBE-1)
		else:
			if normMethod=="AB":
				# Normalise to focus on AB ratio
				newBaseMID = fullMIDCalc(alp,alp/normAB)
				apn,bpn= model(alp,alp/normAB,LET)
				refMID =  fullMIDCalc(apn,bpn)
				refRBE = newBaseMID/refMID
				RBE = (RBE-1)/(refRBE-1)
			else:
				print("Normalisation not recognised, returning un-normalised")

	return RBE

def plotRandomRBECorrelation(normalise= False, normMethod="LET"):
	# Set up some values for plotting
	# Generate a custom diverging colormap
	theseCols = sns.diverging_palette(10, 220, n=10, as_cmap=True)

	# Generate a mask for the lower triangle
	mask = np.zeros_like(np.zeros([len(names),len(names)]), dtype=np.bool)
	mask[np.tril_indices_from(mask)] = True

	# Actually do calculations
	samples = 5000
	alpha = np.random.normal(0.2,0.04,samples)
	aOverB = np.random.uniform(2,10,samples)
	beta = np.array([a/b for a,b in zip(alpha,aOverB)])
	LET = np.random.uniform(1,15,samples)

	# Build table of RBE values
	modelOutputs = []

	for alp,bet,L in zip(alpha,beta,LET):
		modelOutputs.append([])
		for n,mod in enumerate(modelList):
			RBE = calculateRBE(mod,alp, bet, L, normalise, normMethod)
			modelOutputs[-1].append(RBE)

	# Calculate average relative RBEs between models
	relativeRBEs = True
	relEffects = np.zeros([len(names),len(names)])
	if relativeRBEs:
		for i in range(len(modelOutputs[0])):
			relEffects[i][i]=1.0
			for j in range(i+1,len(modelOutputs[0])):
				mod1 = [m[i] for m in modelOutputs]
				mod2 = [m[j] for m in modelOutputs]
				ratios = [m1/m2 for m1, m2 in zip(mod1,mod2)]

				relEffects[i][j]=np.mean(ratios)
				relEffects[j][i]=np.mean(ratios)

	d = pd.DataFrame(data=modelOutputs,columns=names)
	corr = d.corr(method=correlationType) 
	print(corr)

	# Build scatter plots
	pairGrid = sns.pairplot(d, plot_kws={"edgecolor":"face", "alpha": 0.4, "s":8} )
	pairGrid.map_diag(hideCurrentAxis)

	for n in range(len(pairGrid.axes)):
		theAx = pairGrid.axes[n,n]
		theAx.set_axis_off()
		xlims = theAx.get_xlim()
		ylims = theAx.get_ylim()
		theAx.text( (xlims[1]+xlims[0])/2,(ylims[1]+ylims[0])/2,names[n],
					rotation=30, horizontalalignment="center",
					verticalalignment = "center", size=32)

	# Add text labels to diagonal for each model
	for i in range(len(pairGrid.axes)):
		currValue = pairGrid.axes[i,0].get_ylabel()
		pairGrid.axes[i,0].set_ylabel(currValue, rotation=0, horizontalalignment="right", size=32)

		currValue = pairGrid.axes[-1,i].get_xlabel()
		pairGrid.axes[-1,i].set_xlabel(currValue, rotation=90, size=32)

	# Place coloured squares in upper quadrant for correlations
	plt.tight_layout()
	minCorr = False
	if min(corr.min())<0:
		minCorr=True
	for j in range(len(pairGrid.axes)):
		for i in range(j+1,len(pairGrid.axes)):
			theAx = pairGrid.axes[j,i]
			theAx.set_axis_off()

			xlims = theAx.get_xlim()
			ylims = theAx.get_ylim()

			corrVal = corr.iloc[[j],[i]]
			corrVal = (corrVal.values[0])[0]
			if minCorr:
				colour = theseCols((corrVal+1)/2)
			else:
				colour = theseCols((corrVal-min(corr.min()))/(1-min(corr.min())))
			theRect = ptch.Rectangle( (xlims[0],ylims[0]),
									  (xlims[1]-xlims[0]), (ylims[1]-ylims[0]),
									  color=colour)
			theAx.add_patch(theRect)

			textCol = colorForText(colour)
			theAx.text((xlims[1]+xlims[0])/2,(ylims[1]+ylims[0])/2,
					   round(corrVal,2), horizontalalignment="center",
					   verticalalignment = "center", size=42, color=textCol)

	# Hide numbers on axis for scatter plots
	for j in range(len(pairGrid.axes)):
		for i in range(j+1,len(pairGrid.axes)):
			theAx = pairGrid.axes[i,j]
			theAx.get_xaxis().set_visible(False)
			theAx.get_yaxis().set_visible(False)

	# Strip off axis labels from edges
	for i in range(len(pairGrid.axes)):
		pairGrid.axes[i,0].set_ylabel("")
		pairGrid.axes[-1,i].set_xlabel("")

	plt.tight_layout()
	title = "PairPlot"
	if normalise:
		title = title+ " normalised "+normMethod
	plt.savefig(title+".png")
	plt.ioff()
	#plt.show()


plotRandomRBECorrelation(normalise=False)
plotRandomRBECorrelation(normalise=True, normMethod = "LET")
plotRandomRBECorrelation(normalise=True, normMethod = "AB")
