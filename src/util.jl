function formatandjoin(v::AbstractVector)::String
    vstrvec = []
    for vi in v
        push!(vstrvec, @sprintf("%.16e", vi))
    end

    join(vstrvec, ' ')
end

# I can implement a variable space version
function writedat(file::String, columns::AbstractVector...)
    io = open(file, "w")
    try
        for line = zip(columns...)
            linevec = collect(line)
            linestring = formatandjoin(linevec)
            write(io, linestring*'\n')
        end
    finally
        close(io)
    end
end

function get_monotonic_filter(vecs::AbstractVector...)
    n = length(vecs[1])

    mask = trues(n)

    for vec in vecs
        for i in 2:n
            if vec[i] < vec[i-1]
                mask[i] = false
            end
        end
    end

    return mask
end

function filter_monotonic(vecs::AbstractVector...)
    mask = get_monotonic_filter(vecs...)
    filtered = []
    for vec in vecs
        push!(filtered, vec[mask])
    end     

    return filtered
end