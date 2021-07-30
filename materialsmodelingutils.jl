#------------------Energy Potentials----------------------------
#define Lennard-Jones energy potential
lj(r) = 4*(r^-12 - r^-6)

function lj_cutoff(cutoff)
	c2 = (12*(cutoff^-13) - 6*(cutoff^-7))/(2*cutoff)
	c0 = (cutoff^-6) - (cutoff^-12) - c2*(cutoff^2)
	function f(r)
		a = r^-6
		4*(a^2 - a + c2*r^2 + c0)
	end
end

function lj_r2(r2)
	a = r2^-3
	4*(a^2 - a)
end

#we can calculate the pair potential in terms of r^2 instead of r since this will save computation time
function lj_cutoff_r2(cutoff)
	if cutoff == Inf
		lj_r2
	else
		c2 = (12*(cutoff^-13) - 6*(cutoff^-7))/(2*cutoff)
		c0 = (cutoff^-6) - (cutoff^-12) - c2*(cutoff^2)
		function f(r2)
			a = r2^-3
			4*(a^2 - a + c2*r2 + c0)
		end
	end
end

#---------------------Distance Functions----------------------------------
#calculates the squared distance between two points with cartesian coordinate locations.  l is the size of the cubic domain in every direction which we assume is periodically repeated.  This is important because the distance between two points in each direction is the minimum between the distance within the cell and the distance to the periodic extension.
calcr2(p1, p2, ls) = mapreduce(+, p1, p2, ls) do a, b, c
	d = abs(a - b)
	d2 = min(d, abs(c - d))^2
end

#calculates the r^2 contribution of a single coordinate accounting for periodic boundary conditions
function calcr2comp(r_comp, l)
	rabs = abs(r_comp)
	if rabs > l/2
		(rabs - l)^2
	else
		rabs^2
	end
end

# function calcr2(r_comps, l)
# 	r2 = 0.0
# 	for comp in r_comps
# 		r2 += calcr2comp(comp, l)
# 	end
# 	return r2
# end

calcr2s(r_comps, l) = mapreduce((r_comp_list, ldim) -> calcr2comp.(r_comp_list, ldim), (v1, v2) -> v1 .+ v2, r_comps, l)

#calculates the radial distribution function given a list of every pair of distances and the periodic cell size in a single dimension
function calc_3drdf(rs, l, w, h, ntotal, nbins = 100)
	bins = LinRange(0, min(l, w, h)/2, nbins)
	hist = fit(Histogram, rs, bins)
	edges = hist.edges[1]
	binpoints = (edges[2:end] .+ edges[1:end-1]) ./ 2
	weights = hist.weights
	vol = l*w*h
	ρavg = ntotal/vol
	dr = edges[2]-edges[1]
	weights = hist.weights
	expectedcount = (binpoints .^2) .* (4*pi*dr*ρavg*(ntotal-1)/2)
	# dr = bins[2] - bins[1]
	# dv(r) = 4*pi*dr*(r^2)
	# #average particle density
	# ρavg = 2*ntotal/(l^3)
	# weights = 2 ./ ((rs.^2) .* (4*pi*dr*ntotal*(ntotal-1)/(l^3)))
	# fig = histogram(rs, bins = bins, weights = weights, legend = false)
	g = weights ./ expectedcount
	return (binpoints, g, ρavg)
end

function calc_2drdf(rs, l, w, ntotal, nbins)
	# bins = LinRange(0, maximum(rs), nbins)
	# dr = bins[2] - bins[1]
	# dv(r) = 4*pi*dr*(r^2)
	# #average particle density
	# ρavg = 2*ntotal/(l^3)
	# weights = 2 ./ (rs .* (2*pi*dr*ntotal*(ntotal-1)/vol))
	bins = LinRange(0, min(l, w)/2, nbins)
	hist = fit(Histogram, rs, bins)
	vol = l*w
	edges = hist.edges[1]
	binpoints = (edges[2:end] .+ edges[1:end-1]) ./ 2
	dr = edges[2]-edges[1]
	weights = hist.weights
	ρavg = ntotal/vol
	expectedcount = binpoints .* (2*pi*dr*ρavg*(ntotal-1)/2)
	# f = 2 ./ (binpoints .* (2*pi*dr*ntotal*(ntotal-1)/vol))
	g = weights ./ expectedcount
	# histogram(rs, bins = bins, weights = weights, legend=false, xticks=0:maximum(rs))
	return (binpoints, g, ρavg)
end

function calc_1drdf(rs, l, ntotal; nbins = 100)
	bins = LinRange(0, min(l/2, maximum(rs)), nbins)
	dr = bins[2] - bins[1]
	# dv(r) = 4*pi*dr*(r^2)
	# #average particle density
	# ρavg = 2*ntotal/(l^3)
	weights = 2 ./ (ones(length(rs)) .* (dr*ntotal*(ntotal-1)/(l)))
	histogram(rs, bins = bins, weights = weights)
end

function calc_coordination_number(r, g, firstmin, ρavg, dims)
	iend = findfirst(r .>= firstmin)
	dr = r[2] - r[1]
	f(r) = if dims == 2
		2*pi*r
	elseif dims == 3
		4*pi*(r^2)
	else
		1.0
	end
	(dr*ρavg)*sum(f.(r[1:iend]) .* g[1:iend])
end



#----------------------Total Energy Calculation-------------------------
#In the most naive calculation need to calculate the interaction between every point pair.
#This function calculates the distances and energies of every pair from scratch each time.
function calctotalenergy(points, ls)
	#generator for every point pair
	pointpairs = combinations(points, 2)
	mapreduce(pair -> lj_r2(calcr2(pair[1], pair[2], ls)), +, pointpairs)
end

#Alternatively we can save down the state of the system in terms of all the pair distances and then calculate the energy from the existing structure
calctotalenergy(r2s) = sum(r2 -> lj_r2(r2), r2s)


#------------------------System State Calculation and Saving-------------------
#If we change a single coordinate of a single atom at a time then we need a way of updating the pair distances and energies that such a change contributes to without affecting everything else.  We can keep the previous value and ideally just calculate a delta term to modify the calculation.  The most straight forward and memory inefficient way to do this is to keep in memory the x, y, and z contributions of each pair distance calculation.  We also need to indentify from the pair list all of the pairs that include the atom being moved.
function generateatompairlookup(points)
	pairlist = combinations(1:length(points), 2)
	#create blank lookup list for pairs
	pairlookup1 = Dict{Int64, Vector{Int64}}()
	pairlookup2 = Dict{Int64, Vector{Int64}}()
	for i in eachindex(points)
		push!(pairlookup1, i => Vector{Int64}())
		push!(pairlookup2, i => Vector{Int64}())
	end
	#for each point save all the pair indices that it affects both where it appears in the first and second position
	for (i, p) in enumerate(pairlist)
		push!(pairlookup1[p[1]], i)
		push!(pairlookup2[p[2]], i)
	end
	return pairlookup1, pairlookup2
end

#save the x, y, and z components of distances before period adjustments
# generatepairdistances(points) = [[p[1][j] - p[2][j] for j in eachindex(points[1])] for p in combinations(points, 2)]
#create at tuple of vectors each containing that coordinate for every point
generatepairdistances(points) = map(j -> [p[1][j] - p[2][j] for p in combinations(points, 2)], Tuple(eachindex(points[1])))

function generatesystemstate(points, ls; cutoff=Inf)
	points_new = deepcopy(points)
	pairlookup1, pairlookup2 = generateatompairlookup(points)
	r_comps = generatepairdistances(points)
	r_comps_new = deepcopy(r_comps)
	r2s = calcr2s(r_comps, ls)
	r2s_new = deepcopy(r2s)
	f_energy = lj_cutoff_r2(cutoff)
	energies = f_energy.(r2s)
	energies_new = deepcopy(energies)
	return (points, points_new, pairlookup1, pairlookup2, r_comps, r_comps_new, r2s, r2s_new, energies, energies_new, cutoff^2, f_energy)
end

#----------------------System Modification and Update--------------------------------
#update all system information after changing a single point a single coordinate
function candidatestep!(points, points_new, pairlookup1, pairlookup2, r_comps, r_comps_new, r2s, r2s_new, energies, energies_new, cutoffsq, f_energy, pointindex, coordinate, delta, ls)
	#generate list of all distances that have been updated

	#point update
	c = points[pointindex][coordinate]
	cnew = c + delta
	#deltamod keeps track of the actual displacement accounting for period wrapping
	if cnew > ls[coordinate]
		points_new[pointindex][coordinate] = cnew - ls[coordinate]
		deltamod = -ls[coordinate]
	elseif cnew < 0
		points_new[pointindex][coordinate] = cnew + ls[coordinate]
		deltamod = ls[coordinate]
	else
		points_new[pointindex][coordinate] = cnew
		deltamod = 0
	end

	# println("cnew = $cnew, deltamod = $deltamod")

	#accumulate the change in energy for each pair
	energydelta = 0.0

	#update distance components for pairs in which the changed point appears first
	for i in pairlookup1[pointindex]
		d = r_comps[coordinate][i]
		#if p1 is listed first then the distance is p1 - p2 so any change to p1 should be added to the distance
		dnew = d + (delta + deltamod)
		# println("For pair $i, d = $d and dnew = $dnew")
		r_comps_new[coordinate][i] = dnew
		oldr2 = calcr2comp(d, ls[coordinate])
		newr2 = calcr2comp(dnew, ls[coordinate])
		# println("For pair $i, oldr2 = $oldr2 and newr2 = $newr2")
		#update distance of this pair from the displacement in one coordinate
		r2s_new[i] = r2s[i] + (newr2 - oldr2)
		# energies_new[i] = lj_r2(r2s_new[i])
		# energydelta += energies_new[i] - energies[i]
	end

	#update distance components for pairs in which the changed point appears second
	for i in pairlookup2[pointindex]
		d = r_comps[coordinate][i]
		#if p1 is listed second then the distance is p2 - p1 so any change to p1 should be subtracted to the distance
		dnew = d - (delta + deltamod)
		r_comps_new[coordinate][i] = dnew
		oldr2 = calcr2comp(d, ls[coordinate])
		newr2 = calcr2comp(dnew, ls[coordinate])
		#update distance of this pair from the displacement in one coordinate
		r2s_new[i] = r2s[i] + (newr2 - oldr2)
		# energies_new[i] = lj_r2(r2s_new[i])
		# energydelta += energies_new[i] - energies[i]
	end

	for indices in (pairlookup1[pointindex], pairlookup2[pointindex])
		if cutoffsq == Inf
			@inbounds @fastmath @simd for i in indices
				energies_new[i] = f_energy(r2s_new[i])
				energydelta += energies_new[i] - energies[i]
			end
		else
			@inbounds @fastmath for i in indices
				energies_new[i] = r2s_new[i] > cutoffsq ? 0.0 : f_energy(r2s_new[i])
				energydelta += energies_new[i] - energies[i]
			end
		end
	end

	return energydelta
end

function acceptstep!(points, points_new, pairlookup1, pairlookup2, r_comps, r_comps_new, r2s, r2s_new, energies, energies_new, cutoffsq, f_energy, pointindex, coordinate)
	points[pointindex][coordinate] = points_new[pointindex][coordinate]

	for i in pairlookup1[pointindex]
		r_comps[coordinate][i] = r_comps_new[coordinate][i]
		r2s[i] = r2s_new[i]
		energies[i] = energies_new[i]
	end

	for i in pairlookup2[pointindex]
		r_comps[coordinate][i] = r_comps_new[coordinate][i]
		r2s[i] = r2s_new[i]
		energies[i] = energies_new[i]
	end
end

function gibbs_step!(state, ls, maxdelt, temp, stepmode=false)
	acceptcount = 0
	stepcount = 0
	dims = length(state[1][1])
	for i in eachindex(state[1])
		for d in eachindex(state[1][1])
			#creates a displacement uniformly sampled from -maxdelt to maxdelt
			delt = 2*maxdelt*rand() - maxdelt
			energydelta = candidatestep!(state..., i, d, delt, ls)
			
			accept = if energydelta <= 0
				true
			else
				exp(-energydelta/temp) >= rand()
			end
			if accept
				acceptstep!(state..., i, d)
				acceptcount += 1
			end
			if stepmode
				println("Total energy is $(sum(state[9]))")
				println("Displacing point $i and coordinate $d by $delt")
				println("Change in energy is $energydelta")
				if dims == 1
					v = [(a[1], 0.0) for a in state[1]]
					fig = scatter(v, legend=false)
				elseif dims == 2
					fig = scatter(Tuple.(state[1]), legend = false)
				elseif dims == 3
					fig = scatter3d(state[1], legend=false)
				end
				display(fig)
				println("Try next dimension?")
				readline()
			end
			stepcount += 1
		end
	end
	return acceptcount/stepcount
end