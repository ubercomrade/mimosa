# Minimal NPY reader for loading frozen compatibility fixtures without NumPy.
# Supports little-endian float32/float64 and int8/int64/bool arrays.

module NPYReader

struct NPYHeader
    dtype::String
    shape::Vector{Int}
    fortran_order::Bool
    n_bytes::Int
end

const MAGIC = UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]

function read_header(io::IO)
    magic = read(io, 6)
    if magic != MAGIC
        error("not a valid NPY file (bad magic).")
    end
    major = read(io, UInt8)
    minor = read(io, UInt8)
    if major == 1
        header_len = reinterpret(Int16, read(io, 2))[1]
    elseif major == 2
        header_len = reinterpret(Int32, read(io, 4))[1]
    else
        error("unsupported NPY major version: $major")
    end
    header_str = String(read(io, header_len))
    return _parse_header(header_str)
end

function _parse_header(s::AbstractString)
    dict_str = strip(s)
    if startswith(dict_str, "{") && endswith(dict_str, "}")
        dict_str = dict_str[2:(end - 1)]
    end
    parts = _split_top_level(dict_str, ',')
    fields = Dict{String,String}()
    for p in parts
        kv = _split_top_level(p, ':')
        length(kv) == 2 || continue
        key = strip(kv[1])
        if startswith(key, "'") || startswith(key, "\"")
            key = key[2:(end - 1)]
        end
        val = strip(kv[2])
        fields[key] = val
    end
    dtype = _parse_dtype(fields["descr"])
    shape = _parse_shape(fields["shape"])
    fortran_order = haskey(fields, "fortran_order") ? strip(fields["fortran_order"]) == "True" : false
    n_elements = prod(shape; init=1)
    n_bytes = n_elements * _dtype_size(dtype)
    return NPYHeader(dtype, shape, fortran_order, n_bytes)
end

function _split_top_level(s::AbstractString, sep::Char)
    parts = String[]
    depth = 0
    in_quote = false
    quote_char = ' '
    current = IOBuffer()
    for c in s
        if in_quote
            write(current, c)
            if c == quote_char
                in_quote = false
            end
        elseif c in ('\'', '"')
            in_quote = true
            quote_char = c
            write(current, c)
        elseif c == '(' || c == '[' || c == '{'
            depth += 1
            write(current, c)
        elseif c == ')' || c == ']' || c == '}'
            depth -= 1
            write(current, c)
        elseif c == sep && depth == 0
            push!(parts, String(take!(current)))
        else
            write(current, c)
        end
    end
    push!(parts, String(take!(current)))
    return parts
end

function _parse_dtype(descr::AbstractString)
    s = strip(descr)
    if startswith(s, "'") || startswith(s, "\"")
        s = s[2:(end - 1)]
    end
    return s
end

function _dtype_size(dtype::AbstractString)
    endswith(dtype, "i1") && return 1
    endswith(dtype, "b1") && return 1
    endswith(dtype, "i2") && return 2
    endswith(dtype, "f2") && return 2
    endswith(dtype, "i4") && return 4
    endswith(dtype, "f4") && return 4
    endswith(dtype, "i8") && return 8
    endswith(dtype, "f8") && return 8
    error("unsupported dtype: $dtype")
end

function _julia_type(dtype::AbstractString)
    endswith(dtype, "i1") && return Int8
    endswith(dtype, "b1") && return Bool
    endswith(dtype, "i2") && return Int16
    endswith(dtype, "f2") && return Float16
    endswith(dtype, "i4") && return Int32
    endswith(dtype, "f4") && return Float32
    endswith(dtype, "i8") && return Int64
    endswith(dtype, "f8") && return Float64
    error("unsupported dtype: $dtype")
end

function read_npy(path::AbstractString)
    open(path, "r") do io
        header = read_header(io)
        raw = read(io, header.n_bytes)
        T = _julia_type(header.dtype)
        n = div(header.n_bytes, _dtype_size(header.dtype))
        data = Vector{T}(undef, n)
        for i in 1:n
            bytes = raw[(i - 1) * _dtype_size(header.dtype) + 1:i * _dtype_size(header.dtype)]
            data[i] = _read_le(T, bytes)
        end
        if length(header.shape) == 0
            return data[1]
        end
        if header.fortran_order
            return reshape(data, Tuple(header.shape))
        else
            # C order: NumPy stores row-major. reshape to reversed shape gives
            # the transpose, then permutedims reverses back to original layout.
            D = length(header.shape)
            return permutedims(reshape(data, Tuple(reverse(header.shape))), Tuple(D:-1:1))
        end
    end
end

function _read_le(::Type{T}, bytes::Vector{UInt8}) where {T}
    return reinterpret(T, bytes)[1]
end

end # module