### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 8c7b74d3-46c8-4773-a605-d9ac5e033552
begin 
	using Random
	using Distributions
	using Combinatorics
	using Base.Threads
	using StatsBase
	using Statistics
end

# ╔═╡ 1e2f24e0-6948-11eb-2eef-853381ab85f4
using Plots

# ╔═╡ 6fa40d80-694e-11eb-1900-85173ab7e09b
using FileIO

# ╔═╡ d819c511-bc8f-4279-88e3-83572f123443
using ImageIO

# ╔═╡ 677fcd10-6949-11eb-28d4-7961b662a4c7
#tricks to make gibbs sampling work, perturb one coordinate of one atom at a time.  Only calculate incremental change in energy from that coordinate.  Use cutoff potential and do not calculate energy function if distance is greater than cutoff.
include("materialsmodelingutils.jl")

# ╔═╡ 612bbbf0-6948-11eb-2404-d1ce09300d7f
function lj(r)
	a = r^-6
	4*(a^2 - a)
end

# ╔═╡ 6e4c1300-6c8d-11eb-39fa-15d35693ee91
function lj_cutoff(cutoff)
	c2 = (12*(cutoff^-13) - 6*(cutoff^-7))/(2*cutoff)
	c0 = (cutoff^-6) - (cutoff^-12) - c2*(cutoff^2)
	function f(r)
		a = r^-6
		4*(a^2 - a + c2*r^2 + c0)
	end
end

# ╔═╡ 615e8bc0-6948-11eb-1dc8-65bd769ab7ae
r = LinRange(0.5, 3, 10000)

# ╔═╡ 616123d0-6948-11eb-1310-1b507c060915
plotly()

# ╔═╡ 32b26860-6cb7-11eb-344c-038bf7b8b4fd
plot(r, [lj.(r) mapreduce(c -> lj_cutoff(c).(r), hcat, 1.5:0.5:3.0)], yaxis = [-1, 1], lab=["LJ" mapreduce(c -> "LJ Cutoff = $c", hcat, 1.5:0.5:3.0)])

# ╔═╡ cab4a410-6948-11eb-1e5c-2b17564d814d
function createpoints(nl, dims, vol_spacing)
	l = nl*vol_spacing
	spacing = (vol_spacing/2):vol_spacing:l
	evenpoints = Vector{Vector{Float64}}()
	if dims == 1
		for x in spacing
			push!(evenpoints, [x])
		end
	elseif dims == 2
		for x in spacing
			for y in spacing
				push!(evenpoints, [x, y])
			end
		end
	elseif dims == 3
		for x in spacing
			for y in spacing
				for z in spacing 
					push!(evenpoints, [x, y, z])
				end
			end
		end
	end
	return evenpoints, l
end

# ╔═╡ 1bafb080-6949-11eb-3f40-97c089456720
evenpoints3d, l3d= createpoints(4, 3, 3.0)

# ╔═╡ 24f299a2-6949-11eb-13b1-f59bb1f7279b
evenpoints2d, l2d = createpoints(4, 2, 3.0)

# ╔═╡ 09fd5ae0-6949-11eb-1420-71a05dc6acc7
plotly()

# ╔═╡ d419a940-694a-11eb-14fc-31d01c342cc0
scatterpoints2d(points, l) = scatter(Tuple.(points), xaxis = [0, l], yaxis = [0, l], yticks = 0:l, xticks = 0:l, size = (600, 600), legend = false)

# ╔═╡ 7a9c6ed0-694e-11eb-0549-1b58865c07ab
path = download("https://web.northeastern.edu/afeiguin/p4840/p131spring04/img331.png")

# ╔═╡ 85bf6290-694e-11eb-3ad1-813fc622b5aa
img = FileIO.load(path)

# ╔═╡ 8bae3eb0-694e-11eb-22e7-6b73618f9310
plot(img)

# ╔═╡ 0d71cbb0-694a-11eb-0f16-1bbc1d03ac7e
scatterpoints2d(evenpoints2d, l2d)

# ╔═╡ 132c6de0-6949-11eb-205e-07061ee8d8e2
scatter3d(Tuple.(evenpoints3d), xaxis = [0, l3d], yaxis = [0, l3d], zaxis = [0,l3d], yticks = 0:l3d, xticks = 0:l3d, zticks = 0:l3d, size = (600, 600, 600), legend = false)

# ╔═╡ 6391df80-6c06-11eb-2491-1f1257e47dda
function createhexagonalpoints(a, nx::Int64, ny::Int64)
#create an fcc lattice intended to be used with periodic boundary conditions where n's are the number of cells
#in each direction
	basepoints = [	[0.0, 0.0],
					[0.5, sqrt(3)/2] ]
	#to fill space with these points need to translate over in the x direction by 1
	#and in the y direction by sqrt(3)

	#define unit points in terms of a = 1
	#height should be sqrt(3) and width should be 2
	translatex(p, n) = [p[1] + n, p[2]]
	translatey(p, n) = [p[1], p[2]+ n*sqrt(3)]
	xrepeats = nx == 0 ? [] : mapreduce(n -> translatex.(basepoints, n), vcat, 1:nx)
	yrepeats = ny == 0 ? [] : mapreduce(n -> translatey.(vcat(basepoints, xrepeats), n), vcat, 1:ny)
	points = a.*vcat(basepoints, xrepeats, yrepeats)
	cellwidth = a*(nx+1)
	cellheight = a*sqrt(3)*(ny+1)
	return points, cellwidth, cellheight
end

# ╔═╡ 72f466d0-6c08-11eb-1110-81876a69a92d
function show2dgrid(points, a, cellw, cellh, maxdim=600)
	if cellw > cellh
		w=maxdim
		h=maxdim*cellh/cellw
	else
		h = maxdim
		w = maxdim*cellw/cellh
	end
	plt = scatter(Tuple.(points), xaxis = [0, cellw], yaxis=[0,cellh], xticks = 0:a:cellw, yticks = 0:a:cellh, size=(w, h), legend=false, title="Cell Width = $cellw, cell height = $cellh, a = $a")
end

# ╔═╡ 6a14be42-6c06-11eb-101a-8d1cd913578d
a = 1.0; nx = 15; ny = 8

# ╔═╡ 18c86810-6c07-11eb-35e8-718c935b20e4
(hexpoints, hexw, hexh) = createhexagonalpoints(a, nx, ny)

# ╔═╡ ce7a0980-6c06-11eb-3f3c-51f2a62d7fd7
show2dgrid(hexpoints, a, hexw, hexh)

# ╔═╡ a2356ac0-6c08-11eb-0045-712e237cc7a4
#average density
length(hexpoints)/(hexw*hexh)

# ╔═╡ da2431d0-6c0a-11eb-2287-672e63d298c1
hexstate = generatesystemstate(hexpoints, (hexw, hexh))

# ╔═╡ 1e50ef00-6c0c-11eb-31ad-8121f7a56b35
hexbinpoints, hexg, ρavg = calc_2drdf(sqrt.(hexstate[7]), hexh, hexw, length(hexstate[1]), 200)

# ╔═╡ 5c22b4e0-6c1a-11eb-3d0d-6fd1116923f1
plot(hexbinpoints, hexg, legend=false)

# ╔═╡ e5afdb00-6c21-11eb-2105-f1d902b1aba5
calc_coordination_number(hexbinpoints, hexg, 1.2, ρavg, 2)

# ╔═╡ 287b08f2-6cbe-11eb-12da-7508a3ff12b5
function createsquarepoints(nls, a)
	spacings = [0.0:a:(nl-1)*a for nl in nls]
	dims = length(nls)
	evenpoints = Vector{Vector{Float64}}()
	if dims == 1
		for x in spacings[1]
			push!(evenpoints, [x])
		end
	elseif dims == 2
		for x in spacings[1]
			for y in spacings[2]
				push!(evenpoints, [x, y])
			end
		end
	elseif dims == 3
		for x in spacings[1]
			for y in spacings[2]
				for z in spacing[3] 
					push!(evenpoints, [x, y, z])
				end
			end
		end
	end
	ls = Tuple([nl*a+a for nl in nls])
	return evenpoints, ls
end

# ╔═╡ 35ba7410-6cbe-11eb-0d19-8f6eae5db6e1
(squarepoints, ls_square) = createsquarepoints((16, 16), 1.0)

# ╔═╡ 6c544460-6cbe-11eb-24fc-774eef92d587
squaredensity_2d = length(squarepoints)/prod(ls_square)

# ╔═╡ 80935f5e-6cbe-11eb-36c5-5da1089532e5
show2dgrid(squarepoints, a, ls_square...)

# ╔═╡ 2445a730-6cbf-11eb-1c9f-230119848f01
squarestate = generatesystemstate(squarepoints, ls_square)

# ╔═╡ 34b76680-6cbf-11eb-2373-0d67189c2988
squarebinpoints, squareg, ρavg_square = calc_2drdf(sqrt.(squarestate[7]), ls_square..., length(squarestate[1]), 100)

# ╔═╡ 4ee4e872-6cbf-11eb-1e95-2bd4ac006bac
plot(squarebinpoints, squareg, legend=false)

# ╔═╡ 604f5fa0-6cbf-11eb-36a3-45d80ad9e052
#see for the square grid the coordination number is 4
calc_coordination_number(squarebinpoints, squareg, 1.2, ρavg_square, 2)

# ╔═╡ a5174ca0-6c16-11eb-1348-f975e76e929e
function createrandompoints(ntotal, ndims, ls)
#create ntotal points randomly distributed in a volume of ndims dimensions where ls is a tuple of lengths for each dimension
	if length(ls) != ndims
		error("The specified number of dimensions $ndims doesn't match the number of lengths given for the computational cell")
	end
	coordinates = [l.*rand(ntotal) for l in ls]
	points = [[coordinates[j][i] for j in 1:ndims] for i in 1:ntotal]
	return points, ls...
end 

# ╔═╡ aa9fece0-6c16-11eb-3862-85481c841f02
randompoints = createrandompoints(1024, 2, (32, 32))[1]

# ╔═╡ f7350720-6c16-11eb-1e62-e15b8610c159
show2dgrid(randompoints, a, 32, 32)

# ╔═╡ 0a1a3770-6c17-11eb-2550-c9abc4aa09f2
randomstate = generatesystemstate(randompoints, (32, 32))

# ╔═╡ 16bffa9e-6c17-11eb-2593-456abc4ffbf1
randombinpoints, randg, ρavgrand = calc_2drdf(sqrt.(randomstate[7]), 32, 32, length(randompoints), 100)

# ╔═╡ 6fa67580-6c22-11eb-1864-23b819ddd45e
bar(randombinpoints, randg, legend=false)

# ╔═╡ 83da71b0-6c8a-11eb-1420-1b3ffdc6dea8
function findequilibrium(state0, ls, maxsteps, temps)
	r2srecord = Vector{typeof(state0[7])}(undef, maxsteps)
	pointrecord = Vector{typeof(state0[1])}(undef, maxsteps)
	energyrecord = Vector{Float64}(undef, maxsteps)
	maxdelt = 0.1
	n = 0
	dims = length(state0[1][1])
	state = deepcopy(state0)
	println("Starting energy is $(sum(state[9])) which should equal $(calctotalenergy(state[1], ls))")
	println("Starting accept rate is $r")
	t = time()
	for i in 1:maxsteps
		r = gibbs_step!(state, ls, maxdelt, temps[1])
		if r > 0.25
			maxdelt = min(minimum(ls)/2, maxdelt*1.1)
		else
			maxdelt *= 0.9
		end
		r2srecord[i] = deepcopy(state[7])
		pointrecord[i] = deepcopy(state[1])
		energyrecord[i] = sum(state[9])
	end
	println("Finished after $maxsteps steps, delta range is $maxdelt, accept rate is $r, and total energy is $(sum(state[9]))")
	return pointrecord, r2srecord, energyrecord
end

# ╔═╡ c5b5c4d0-6c8b-11eb-294c-fdcebaf596f2
begin
	ntotal = 256
	area = 512
	ls2d = (sqrt(512), sqrt(512))
	random2dpoints = createrandompoints(ntotal, 2, ls2d)[1]
	random2dstate = generatesystemstate(random2dpoints, ls2d, cutoff=4.0)
end

# ╔═╡ fe60e4f0-6cb7-11eb-1bb3-8b9e81ceb027
plotly()

# ╔═╡ cf32db50-6c8c-11eb-3d98-937555230562
show2dgrid(random2dpoints, a, ls2d...)

# ╔═╡ ed2f9fa0-6c8a-11eb-25ff-51752b1ba544
begin
	numsteps = 10000
	temps = 0.1*ones(numsteps)
	pointrecord, r2srecord, energyrecord = findequilibrium(random2dstate, ls2d, numsteps, temps)
end

# ╔═╡ a0fc8612-6c8b-11eb-2621-b737b60970ea
plot(energyrecord[100:end])

# ╔═╡ 3ed42090-6c8d-11eb-3e35-496079b70943
show2dgrid(pointrecord[end], a, ls2d...)

# ╔═╡ 22864820-6cb8-11eb-0122-1fa6ac054a51
gr()

# ╔═╡ be3a5140-6cb7-11eb-3314-fd8cb8947ba7
#early behavior until equilibrium is reached
@gif for points in pointrecord[1:10:2500]
	show2dgrid(points, a, ls2d...)
end

# ╔═╡ 606276a0-6cb8-11eb-0ede-836d4efd4123
#can compare RDF function to the hexagonal grid from earlier
equilibrium2dbins, equilibrium2dg, ρavgequilibrium2d = calc_2drdf(sqrt.(r2srecord[end]), ls2d..., ntotal, 100)

# ╔═╡ 0e043b40-6cb9-11eb-2b3d-eb84582f8640
plotly()

# ╔═╡ 07d96790-6cb9-11eb-2ec2-1bd183143063
plot(equilibrium2dbins, equilibrium2dg, legend=false)

# ╔═╡ 67921ab0-6cb9-11eb-2502-f1f4eeb48267
#One idea of using Gibbs sampling is to collect statistics across many configurations so we can do this with the RDF
rdfseries_2dequilibrium = [calc_2drdf(sqrt.(r2s), ls2d..., ntotal, 100) for r2s in r2srecord[1:100:end]]

# ╔═╡ a9a89780-6cb9-11eb-0ea4-4bce5cdc5b9a
#we can see the rdf evolve over time to the final configuration
plot(rdfseries_2dequilibrium[1][1], [rdfseries_2dequilibrium[i][2] for i in (1, 10, 100)])

# ╔═╡ 29ade070-6cba-11eb-15db-3b3ed0c72467
#can get a smoother rdf by taking the average values across bins
smoothg_2dequilibrium = sum(a -> a[2], rdfseries_2dequilibrium[25:end]) ./ length(rdfseries_2dequilibrium[25:end])

# ╔═╡ 82d535e0-6cba-11eb-0b12-1bedca5d1534
plot(equilibrium2dbins, [smoothg_2dequilibrium equilibrium2dg], legend=false)

# ╔═╡ bc37c322-6cba-11eb-2837-073cbaf7ee11
#interesting we see the effect of the voids here, if we redo everything with a higher density this value should approach 3
calc_coordination_number(equilibrium2dbins, smoothg_2dequilibrium, 1.65, ρavgequilibrium2d, 2)

# ╔═╡ 02715cc0-6cbb-11eb-1cb7-5371435764ca
#based on the packing fraction of the hexagonal grid and the equilibrium nearest neighbor distance, the fully filled 2d density should be about 0.9565 area units per atom
begin
	square2dpoints, ls2d_square = createsquarepoints((16, 16), 1.0)
	square2dstate_filled = generatesystemstate(square2dpoints, ls2d_square, cutoff=4.0)
end

# ╔═╡ 755cc3f0-6cbb-11eb-3dc2-716b1849d77b
#adjust the number of the temperature factor up and down to see liquid vs solid
begin
	pointrecord_2dfilled, r2srecord_2dfilled, energyrecord_2dfilled = 	findequilibrium(square2dstate_filled, ls2d_square, numsteps, 10 .*temps)
end

# ╔═╡ e5fefe70-6cbb-11eb-3032-9b2a02694621
plot(energyrecord_2dfilled[100:end])

# ╔═╡ ef9e8680-6cbb-11eb-1067-c5db85dca0b5
show2dgrid(pointrecord_2dfilled[end], a, ls2d_square...)

# ╔═╡ 49c99e10-6cc1-11eb-3f95-8f495c2a308d
rdfseries_2dhightemp = [calc_2drdf(sqrt.(r2s), ls2d_square..., ntotal, 100) for r2s in r2srecord_2dfilled[1:100:end]]

# ╔═╡ 7469f0c0-6cc1-11eb-1a48-f1b50a70818f
smoothg_2dhightemp = sum(a -> a[2], rdfseries_2dhightemp[25:end]) ./ length(rdfseries_2dhightemp[25:end])

# ╔═╡ 8eae2f00-6cc1-11eb-187f-4f6d7adc37c7
plot(rdfseries_2dhightemp[1][1], smoothg_2dhightemp, legend=false)

# ╔═╡ b6b0cc60-6cc1-11eb-045d-3f0200e45668
#here with high density the coordination number approaches 6 but at higher temperatures we see the RDF looks more like a liquid than a solid. At very high temperatures the first peak is close to 1 which is the start of the LJ energy wall
calc_coordination_number(rdfseries_2dhightemp[1][1], smoothg_2dhightemp, 1.5, rdfseries_2dhightemp[1][3], 2)

# ╔═╡ bcc1a810-6cc4-11eb-0a34-6ba1917541b8
#3D Points
function createfccpoints(a, nx::Int64, ny::Int64, nz::Int64)
#create an fcc lattice intended to be used with periodic boundary conditions where n's are the number of cells
#in each direction
	basepoints = [	[0.0, 0.0, 0.0],
					[0.5, 0.5, 0.0],
					[0.5, 0.0, 0.5],
					[0.0, 0.5, 0.5] ]
	#to fill space with these points need to translate over in the x, y, and z direction by 1

	#define unit points in terms of a = 1
	#height should be sqrt(3) and width should be 2
	translatex(p, n) = [p[1] + n, p[2], p[3]]
	translatey(p, n) = [p[1], p[2] + n, p[3]]
	translatez(p, n) = [p[1], p[2], p[3] + n]
	xs = mapreduce(n -> translatex.(basepoints, n), vcat, 0:nx)
	ys = ny == 0 ? [] : mapreduce(n -> translatey.(xs, n), vcat, 1:ny)
	zs = nz == 0 ? [] : mapreduce(n -> translatez.(vcat(xs, ys), n), vcat, 1:nz)
	points = a.*vcat(xs, ys, zs)
	w = a*(nx+1)
	l = a*(ny+1)
	h = a*(nz+1)
	return points, (w, l, h)
end


# ╔═╡ d43a1c20-6cc4-11eb-3c03-835984e78fb3
fccpoints, fccls = createfccpoints(1.0, 5, 5, 5)

# ╔═╡ 2777eca0-6cc5-11eb-3c53-9536e6b74c32
function show3dgrid(points, a, ls, maxdim=600)
	plt = scatter3d(Tuple.(points), xaxis = [0, ls[1]], yaxis=[0,ls[2]], zaxis = [0,ls[3]], xticks = 0:a:ls[1], yticks = 0:a:ls[2], zticks=0:a:ls[3], size=(maxdim, maxdim, maxdim), legend=false)
end

# ╔═╡ 7ebd3ba0-6cc5-11eb-2e6e-f1645861351d
show3dgrid(fccpoints, 1.0, fccls)

# ╔═╡ 1c65cc50-6cc6-11eb-37ea-5f8244efe5ef
fccstate = generatesystemstate(fccpoints, fccls)

# ╔═╡ 4bf34290-6cc6-11eb-0cc8-e9122ce2ab0f


# ╔═╡ 37091a80-6cc6-11eb-3aa8-27258125a5f1
fccbinpoints, fccg, ρavg_fcc = calc_3drdf(sqrt.(fccstate[7]), fccls..., length(fccstate[1]), 100)

# ╔═╡ 93365cf0-6cc6-11eb-147a-a5b77b3dbd11
plot(fccbinpoints, fccg, legend=false)

# ╔═╡ c2332bf0-6cc6-11eb-390f-d784b07a57f8
ρavg_fcc

# ╔═╡ a22db140-6cc6-11eb-3ed7-2f2d15f658fe
calc_coordination_number(fccbinpoints, fccg, 0.86, ρavg_fcc, 3)

# ╔═╡ e93cc350-6cc6-11eb-3319-f52f06e8dec1
begin
	nl_3d = 5
	ntotal_3d = 5^3
	vol = ntotal_3d/5.0
	ls3d = (vol^(1/3), vol^(1/3), vol^(1/3))
	random3dpoints = createrandompoints(ntotal_3d, 3, ls3d)[1]
	random3dstate = generatesystemstate(random3dpoints, ls3d, cutoff=4.0)
end

# ╔═╡ 5f1f9890-6cc7-11eb-2fc2-333e84c45e92
show3dgrid(random3dpoints, a, ls3d)

# ╔═╡ 75494580-6cc7-11eb-33dc-f34c70c41881
pointrecord_3d, r2srecord_3d, energyrecord_3d = findequilibrium(random3dstate, ls3d, numsteps, temps)

# ╔═╡ 99dbc260-6cc7-11eb-0e15-3972867aa492
plot(energyrecord_3d[100:end])

# ╔═╡ a2f0e4c0-6cc7-11eb-3d52-936ce6160fb1
show3dgrid(pointrecord_3d[end], a, ls3d)

# ╔═╡ d946ce40-6cc7-11eb-0072-9d7997108b71
gr()

# ╔═╡ b6452a90-6cc7-11eb-2469-dd1e9b0582a7
#early behavior until equilibrium is reached
@gif for points in pointrecord_3d[1:10:2500]
	show3dgrid(points, a, ls3d)
end

# ╔═╡ e053eba0-6cc7-11eb-2720-0140ec9e7c7c
#can compare RDF function to the hexagonal grid from earlier
equilibrium3dbins, equilibrium3dg, ρavgequilibrium3d = calc_3drdf(sqrt.(r2srecord_3d[end]), ls3d..., ntotal_3d, 100)

# ╔═╡ 3876c960-6cc8-11eb-3f39-510bb78c5846
ρavgequilibrium3d

# ╔═╡ 1db9d8ae-6cc8-11eb-3d41-edb1d2d559b4
plotly()

# ╔═╡ 1a325f9e-6cc8-11eb-3780-677ab935880e
#fcc equilibrium density is 4.0*a and the nearest neighbor spacing is a*sqrt(2)/2 so if the spacing should be ~1.07 then we have a = 2*1.07/sqrt(2) = 1.5132 and a density of 6.05 
plot(equilibrium3dbins, equilibrium3dg, legend=false)

# ╔═╡ 35b0d620-6cc9-11eb-13ad-177556a7b4f8
calc_coordination_number(equilibrium3dbins, equilibrium3dg, 0.85, ρavgequilibrium3d, 3)

# ╔═╡ 84a14580-6cc9-11eb-11a3-dfadbd19f6d9
rdfseries_3dequilibrium = [calc_3drdf(sqrt.(r2s), ls3d..., ntotal_3d, 100) for r2s in r2srecord_3d[1:100:end]]

# ╔═╡ 7824b530-6cc9-11eb-3937-6d523d3676fd
smoothg_3dequilibrium = sum(a -> a[2], rdfseries_3dequilibrium[25:end]) ./ length(rdfseries_3dequilibrium[25:end])

# ╔═╡ a5d79f12-6cc9-11eb-0dde-99cd7ed65f80
plot(equilibrium3dbins, smoothg_3dequilibrium, legend=false)

# ╔═╡ bede2d30-6cc9-11eb-252d-a907f9e02785
calc_coordination_number(equilibrium3dbins, smoothg_3dequilibrium, 0.85, ρavgequilibrium3d, 3)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Combinatorics = "~1.0.2"
Distributions = "~0.25.11"
FileIO = "~1.10.1"
ImageIO = "~0.5.6"
Plots = "~1.19.4"
StatsBase = "~0.33.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "ad613c934ec3a3aa0ff19b91f15a16d56ed404b5"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.0.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random", "StaticArrays"]
git-tree-sha1 = "ed268efe58512df8c7e224d2e170afd76dd6a417"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.13.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "42a9b08d3f2f951c9b283ea427d96ed9f1f30343"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.5"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3889f646423ce91dd1055a76317e9a1d3a23fff1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.11"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "256d8e6188f3f1ebfa1a5d17e072a0efafa8c5bf"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.10.1"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f473cdf6e2eb360c576f9822e7c765dd9d26dbc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "eaf96e05a880f3db5ded5a5a8a7817ecba3c7392"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "c6a1fff2fd4b1da29d3dccaffb1e1001244d844e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.12"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "75f7fea2b3601b58f24ee83617b528e57160cbfd"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.1"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "d067570b4d4870a942b19d9ceacaea4fb39b69a1"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.6"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[MappedArrays]]
git-tree-sha1 = "18d3584eebc861e311a552cbb67723af8edff5de"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "4f825c6da64aebaa22cc058ecfceed1ab9af1c7e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.3"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "520e28d4026d16dcf7b8c8140a3041f0e20a9ca8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.7"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fa5e78929aebc3f6b56e1a88cf505bb00a354c4"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.8"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "94bf17e83a0e4b20c8d77f6af8ffe8cc3b386c0a"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.1"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "1e72752052a3893d0f7103fbac728b60b934f5a5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.19.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "508822dca004bf62e210609148511ad03ce8f1d8"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.0"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "885838778bb6f0136f8317757d7803e0d81201e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.9"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "03fb246ac6e6b7cb7abac3b3302447d55b43270e"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.1"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═8c7b74d3-46c8-4773-a605-d9ac5e033552
# ╠═677fcd10-6949-11eb-28d4-7961b662a4c7
# ╠═1e2f24e0-6948-11eb-2eef-853381ab85f4
# ╠═612bbbf0-6948-11eb-2404-d1ce09300d7f
# ╠═6e4c1300-6c8d-11eb-39fa-15d35693ee91
# ╠═615e8bc0-6948-11eb-1dc8-65bd769ab7ae
# ╠═616123d0-6948-11eb-1310-1b507c060915
# ╠═32b26860-6cb7-11eb-344c-038bf7b8b4fd
# ╠═cab4a410-6948-11eb-1e5c-2b17564d814d
# ╠═1bafb080-6949-11eb-3f40-97c089456720
# ╠═24f299a2-6949-11eb-13b1-f59bb1f7279b
# ╠═09fd5ae0-6949-11eb-1420-71a05dc6acc7
# ╠═d419a940-694a-11eb-14fc-31d01c342cc0
# ╠═6fa40d80-694e-11eb-1900-85173ab7e09b
# ╠═d819c511-bc8f-4279-88e3-83572f123443
# ╠═7a9c6ed0-694e-11eb-0549-1b58865c07ab
# ╠═85bf6290-694e-11eb-3ad1-813fc622b5aa
# ╠═8bae3eb0-694e-11eb-22e7-6b73618f9310
# ╠═0d71cbb0-694a-11eb-0f16-1bbc1d03ac7e
# ╠═132c6de0-6949-11eb-205e-07061ee8d8e2
# ╠═6391df80-6c06-11eb-2491-1f1257e47dda
# ╠═72f466d0-6c08-11eb-1110-81876a69a92d
# ╠═6a14be42-6c06-11eb-101a-8d1cd913578d
# ╠═18c86810-6c07-11eb-35e8-718c935b20e4
# ╠═ce7a0980-6c06-11eb-3f3c-51f2a62d7fd7
# ╠═a2356ac0-6c08-11eb-0045-712e237cc7a4
# ╠═da2431d0-6c0a-11eb-2287-672e63d298c1
# ╠═1e50ef00-6c0c-11eb-31ad-8121f7a56b35
# ╠═5c22b4e0-6c1a-11eb-3d0d-6fd1116923f1
# ╠═e5afdb00-6c21-11eb-2105-f1d902b1aba5
# ╠═287b08f2-6cbe-11eb-12da-7508a3ff12b5
# ╠═35ba7410-6cbe-11eb-0d19-8f6eae5db6e1
# ╠═6c544460-6cbe-11eb-24fc-774eef92d587
# ╠═80935f5e-6cbe-11eb-36c5-5da1089532e5
# ╠═2445a730-6cbf-11eb-1c9f-230119848f01
# ╠═34b76680-6cbf-11eb-2373-0d67189c2988
# ╠═4ee4e872-6cbf-11eb-1e95-2bd4ac006bac
# ╠═604f5fa0-6cbf-11eb-36a3-45d80ad9e052
# ╠═a5174ca0-6c16-11eb-1348-f975e76e929e
# ╠═aa9fece0-6c16-11eb-3862-85481c841f02
# ╠═f7350720-6c16-11eb-1e62-e15b8610c159
# ╠═0a1a3770-6c17-11eb-2550-c9abc4aa09f2
# ╠═16bffa9e-6c17-11eb-2593-456abc4ffbf1
# ╠═6fa67580-6c22-11eb-1864-23b819ddd45e
# ╠═83da71b0-6c8a-11eb-1420-1b3ffdc6dea8
# ╠═c5b5c4d0-6c8b-11eb-294c-fdcebaf596f2
# ╠═fe60e4f0-6cb7-11eb-1bb3-8b9e81ceb027
# ╠═cf32db50-6c8c-11eb-3d98-937555230562
# ╠═ed2f9fa0-6c8a-11eb-25ff-51752b1ba544
# ╠═a0fc8612-6c8b-11eb-2621-b737b60970ea
# ╠═3ed42090-6c8d-11eb-3e35-496079b70943
# ╠═22864820-6cb8-11eb-0122-1fa6ac054a51
# ╠═be3a5140-6cb7-11eb-3314-fd8cb8947ba7
# ╠═606276a0-6cb8-11eb-0ede-836d4efd4123
# ╠═0e043b40-6cb9-11eb-2b3d-eb84582f8640
# ╠═07d96790-6cb9-11eb-2ec2-1bd183143063
# ╠═67921ab0-6cb9-11eb-2502-f1f4eeb48267
# ╠═a9a89780-6cb9-11eb-0ea4-4bce5cdc5b9a
# ╠═29ade070-6cba-11eb-15db-3b3ed0c72467
# ╠═82d535e0-6cba-11eb-0b12-1bedca5d1534
# ╠═bc37c322-6cba-11eb-2837-073cbaf7ee11
# ╠═02715cc0-6cbb-11eb-1cb7-5371435764ca
# ╠═755cc3f0-6cbb-11eb-3dc2-716b1849d77b
# ╠═e5fefe70-6cbb-11eb-3032-9b2a02694621
# ╠═ef9e8680-6cbb-11eb-1067-c5db85dca0b5
# ╠═49c99e10-6cc1-11eb-3f95-8f495c2a308d
# ╠═7469f0c0-6cc1-11eb-1a48-f1b50a70818f
# ╠═8eae2f00-6cc1-11eb-187f-4f6d7adc37c7
# ╠═b6b0cc60-6cc1-11eb-045d-3f0200e45668
# ╠═bcc1a810-6cc4-11eb-0a34-6ba1917541b8
# ╠═d43a1c20-6cc4-11eb-3c03-835984e78fb3
# ╠═2777eca0-6cc5-11eb-3c53-9536e6b74c32
# ╠═7ebd3ba0-6cc5-11eb-2e6e-f1645861351d
# ╠═1c65cc50-6cc6-11eb-37ea-5f8244efe5ef
# ╠═4bf34290-6cc6-11eb-0cc8-e9122ce2ab0f
# ╠═37091a80-6cc6-11eb-3aa8-27258125a5f1
# ╠═93365cf0-6cc6-11eb-147a-a5b77b3dbd11
# ╠═c2332bf0-6cc6-11eb-390f-d784b07a57f8
# ╠═a22db140-6cc6-11eb-3ed7-2f2d15f658fe
# ╠═e93cc350-6cc6-11eb-3319-f52f06e8dec1
# ╠═5f1f9890-6cc7-11eb-2fc2-333e84c45e92
# ╠═75494580-6cc7-11eb-33dc-f34c70c41881
# ╠═99dbc260-6cc7-11eb-0e15-3972867aa492
# ╠═a2f0e4c0-6cc7-11eb-3d52-936ce6160fb1
# ╠═d946ce40-6cc7-11eb-0072-9d7997108b71
# ╠═b6452a90-6cc7-11eb-2469-dd1e9b0582a7
# ╠═e053eba0-6cc7-11eb-2720-0140ec9e7c7c
# ╠═3876c960-6cc8-11eb-3f39-510bb78c5846
# ╠═1db9d8ae-6cc8-11eb-3d41-edb1d2d559b4
# ╠═1a325f9e-6cc8-11eb-3780-677ab935880e
# ╠═35b0d620-6cc9-11eb-13ad-177556a7b4f8
# ╠═84a14580-6cc9-11eb-11a3-dfadbd19f6d9
# ╠═7824b530-6cc9-11eb-3937-6d523d3676fd
# ╠═a5d79f12-6cc9-11eb-0dde-99cd7ed65f80
# ╠═bede2d30-6cc9-11eb-252d-a907f9e02785
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
