package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"syscall"
)

// Magic constants
const (
	GGUFMagic   uint32 = 0x46554747 // "GGUF"
	GGUFVersion uint32 = 3
)

// GGMLType identifies the quantization type of a tensor
type GGMLType uint32

const (
	GGMLTypeF32  GGMLType = 0
	GGMLTypeF16  GGMLType = 1
	GGMLTypeQ4_0 GGMLType = 2
	GGMLTypeQ4_1 GGMLType = 3
	GGMLTypeQ5_0 GGMLType = 6
	GGMLTypeQ5_1 GGMLType = 7
	GGMLTypeQ8_0 GGMLType = 8
	GGMLTypeQ8_1 GGMLType = 9
	GGMLTypeQ2_K GGMLType = 10
	GGMLTypeQ3_K GGMLType = 11
	GGMLTypeQ4_K GGMLType = 12
	GGMLTypeQ5_K GGMLType = 13
	GGMLTypeQ6_K GGMLType = 14
	GGMLTypeQ8_K GGMLType = 15
	GGMLTypeBF16 GGMLType = 28
)

func (t GGMLType) String() string {
	switch t {
	case GGMLTypeF32:
		return "F32"
	case GGMLTypeF16:
		return "F16"
	case GGMLTypeQ4_0:
		return "Q4_0"
	case GGMLTypeQ4_1:
		return "Q4_1"
	case GGMLTypeQ5_0:
		return "Q5_0"
	case GGMLTypeQ5_1:
		return "Q5_1"
	case GGMLTypeQ8_0:
		return "Q8_0"
	case GGMLTypeQ4_K:
		return "Q4_K"
	case GGMLTypeQ5_K:
		return "Q5_K"
	case GGMLTypeQ6_K:
		return "Q6_K"
	case GGMLTypeQ8_K:
		return "Q8_K"
	case GGMLTypeBF16:
		return "BF16"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", uint32(t))
	}
}

// MetadataValueType identifies the type of a metadata value
type MetadataValueType uint32

const (
	TypeUint8   MetadataValueType = 0
	TypeInt8    MetadataValueType = 1
	TypeUint16  MetadataValueType = 2
	TypeInt16   MetadataValueType = 3
	TypeUint32  MetadataValueType = 4
	TypeInt32   MetadataValueType = 5
	TypeFloat32 MetadataValueType = 6
	TypeBool    MetadataValueType = 7
	TypeString  MetadataValueType = 8
	TypeArray   MetadataValueType = 9
	TypeUint64  MetadataValueType = 10
	TypeInt64   MetadataValueType = 11
	TypeFloat64 MetadataValueType = 12
)

// TensorInfo describes a tensor in the GGUF file
type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       GGMLType
	Offset     uint64 // Offset relative to tensor_data_offset
	Data       []byte // Mapped slice pointing to tensor payload
}

// NumElements returns total number of elements in tensor
func (t *TensorInfo) NumElements() uint64 {
	if len(t.Dimensions) == 0 {
		return 0
	}
	n := uint64(1)
	for _, d := range t.Dimensions {
		n *= d
	}
	return n
}

// File represents a parsed and mmap'd GGUF file
type File struct {
	Header           Header
	Metadata         map[string]any
	Tensors          []TensorInfo
	TensorMap        map[string]*TensorInfo
	mmapData         []byte
	tensorDataOffset uint64
}

// Header contains GGUF header fields
type Header struct {
	Magic           uint32
	Version         uint32
	TensorCount     uint64
	MetadataKVCount uint64
}

// Open reads and memory-maps a GGUF file
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf open: %w", err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("gguf stat: %w", err)
	}
	fileSize := fi.Size()

	if fileSize < 24 {
		return nil, fmt.Errorf("gguf: file too small (%d bytes)", fileSize)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(fileSize), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("gguf mmap: %w", err)
	}

	gf := &File{
		mmapData:  data,
		Metadata:  make(map[string]any),
		TensorMap: make(map[string]*TensorInfo),
	}

	if err := gf.parse(); err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}

	return gf, nil
}

// Close unmaps the memory
func (gf *File) Close() error {
	if gf.mmapData != nil {
		err := syscall.Munmap(gf.mmapData)
		gf.mmapData = nil
		return err
	}
	return nil
}

type reader struct {
	data []byte
	pos  uint64
}

func (r *reader) readUint8() (uint8, error) {
	if r.pos+1 > uint64(len(r.data)) {
		return 0, io.ErrUnexpectedEOF
	}
	v := r.data[r.pos]
	r.pos++
	return v, nil
}

func (r *reader) readInt8() (int8, error) {
	v, err := r.readUint8()
	return int8(v), err
}

func (r *reader) readUint16() (uint16, error) {
	if r.pos+2 > uint64(len(r.data)) {
		return 0, io.ErrUnexpectedEOF
	}
	v := binary.LittleEndian.Uint16(r.data[r.pos:])
	r.pos += 2
	return v, nil
}

func (r *reader) readInt16() (int16, error) {
	v, err := r.readUint16()
	return int16(v), err
}

func (r *reader) readUint32() (uint32, error) {
	if r.pos+4 > uint64(len(r.data)) {
		return 0, io.ErrUnexpectedEOF
	}
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return v, nil
}

func (r *reader) readInt32() (int32, error) {
	v, err := r.readUint32()
	return int32(v), err
}

func (r *reader) readUint64() (uint64, error) {
	if r.pos+8 > uint64(len(r.data)) {
		return 0, io.ErrUnexpectedEOF
	}
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return v, nil
}

func (r *reader) readInt64() (int64, error) {
	v, err := r.readUint64()
	return int64(v), err
}

func (r *reader) readFloat32() (float32, error) {
	v, err := r.readUint32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(v), nil
}

func (r *reader) readFloat64() (float64, error) {
	v, err := r.readUint64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(v), nil
}

func (r *reader) readBool() (bool, error) {
	v, err := r.readUint8()
	return v != 0, err
}

func (r *reader) readString() (string, error) {
	length, err := r.readUint64()
	if err != nil {
		return "", err
	}
	if r.pos+length > uint64(len(r.data)) {
		return "", io.ErrUnexpectedEOF
	}
	s := string(r.data[r.pos : r.pos+length])
	r.pos += length
	return s, nil
}

func (r *reader) readValue(vt MetadataValueType) (any, error) {
	switch vt {
	case TypeUint8:
		return r.readUint8()
	case TypeInt8:
		return r.readInt8()
	case TypeUint16:
		return r.readUint16()
	case TypeInt16:
		return r.readInt16()
	case TypeUint32:
		return r.readUint32()
	case TypeInt32:
		return r.readInt32()
	case TypeFloat32:
		return r.readFloat32()
	case TypeBool:
		return r.readBool()
	case TypeString:
		return r.readString()
	case TypeUint64:
		return r.readUint64()
	case TypeInt64:
		return r.readInt64()
	case TypeFloat64:
		return r.readFloat64()
	case TypeArray:
		elemTypeVal, err := r.readUint32()
		if err != nil {
			return nil, err
		}
		elemType := MetadataValueType(elemTypeVal)
		count, err := r.readUint64()
		if err != nil {
			return nil, err
		}

		// Specialized fast paths for common arrays
		if elemType == TypeString {
			arr := make([]string, count)
			for i := uint64(0); i < count; i++ {
				s, err := r.readString()
				if err != nil {
					return nil, err
				}
				arr[i] = s
			}
			return arr, nil
		}
		if elemType == TypeFloat32 {
			arr := make([]float32, count)
			for i := uint64(0); i < count; i++ {
				v, err := r.readFloat32()
				if err != nil {
					return nil, err
				}
				arr[i] = v
			}
			return arr, nil
		}
		if elemType == TypeInt32 {
			arr := make([]int32, count)
			for i := uint64(0); i < count; i++ {
				v, err := r.readInt32()
				if err != nil {
					return nil, err
				}
				arr[i] = v
			}
			return arr, nil
		}

		arr := make([]any, count)
		for i := uint64(0); i < count; i++ {
			v, err := r.readValue(elemType)
			if err != nil {
				return nil, err
			}
			arr[i] = v
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown metadata value type: %d", vt)
	}
}

func (gf *File) parse() error {
	r := &reader{data: gf.mmapData, pos: 0}

	magic, err := r.readUint32()
	if err != nil {
		return fmt.Errorf("read magic: %w", err)
	}
	if magic != GGUFMagic {
		return fmt.Errorf("invalid GGUF magic: 0x%08x (expected 0x%08x)", magic, GGUFMagic)
	}
	gf.Header.Magic = magic

	version, err := r.readUint32()
	if err != nil {
		return fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return fmt.Errorf("unsupported GGUF version %d (only v2/v3 supported)", version)
	}
	gf.Header.Version = version

	gf.Header.TensorCount, err = r.readUint64()
	if err != nil {
		return fmt.Errorf("read tensor count: %w", err)
	}

	gf.Header.MetadataKVCount, err = r.readUint64()
	if err != nil {
		return fmt.Errorf("read metadata count: %w", err)
	}

	// Parse metadata
	for i := uint64(0); i < gf.Header.MetadataKVCount; i++ {
		key, err := r.readString()
		if err != nil {
			return fmt.Errorf("metadata[%d] key: %w", i, err)
		}
		valType, err := r.readUint32()
		if err != nil {
			return fmt.Errorf("metadata[%d] valType: %w", i, err)
		}
		val, err := r.readValue(MetadataValueType(valType))
		if err != nil {
			return fmt.Errorf("metadata[%s] value: %w", key, err)
		}
		gf.Metadata[key] = val
	}

	// Alignment
	alignment := uint64(32)
	if val, ok := gf.Metadata["general.alignment"]; ok {
		switch v := val.(type) {
		case uint32:
			alignment = uint64(v)
		case uint64:
			alignment = v
		}
	}

	// Parse tensor info
	gf.Tensors = make([]TensorInfo, gf.Header.TensorCount)
	for i := uint64(0); i < gf.Header.TensorCount; i++ {
		ti := &gf.Tensors[i]
		name, err := r.readString()
		if err != nil {
			return fmt.Errorf("tensor[%d] name: %w", i, err)
		}
		ti.Name = name

		nDims, err := r.readUint32()
		if err != nil {
			return fmt.Errorf("tensor[%s] nDims: %w", name, err)
		}
		ti.Dimensions = make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			dim, err := r.readUint64()
			if err != nil {
				return fmt.Errorf("tensor[%s] dim[%d]: %w", name, d, err)
			}
			ti.Dimensions[d] = dim
		}

		tType, err := r.readUint32()
		if err != nil {
			return fmt.Errorf("tensor[%s] type: %w", name, err)
		}
		ti.Type = GGMLType(tType)

		offset, err := r.readUint64()
		if err != nil {
			return fmt.Errorf("tensor[%s] offset: %w", name, err)
		}
		ti.Offset = offset
		gf.TensorMap[name] = ti
	}

	// Tensor data offset begins after tensor_infos, aligned to general.alignment
	pad := r.pos % alignment
	if pad != 0 {
		r.pos += (alignment - pad)
	}
	gf.tensorDataOffset = r.pos

	// Map data slices for each tensor
	for i := range gf.Tensors {
		ti := &gf.Tensors[i]
		absOffset := gf.tensorDataOffset + ti.Offset
		byteSize := tensorByteSize(ti.Type, ti.Dimensions)
		if absOffset+byteSize > uint64(len(gf.mmapData)) {
			return fmt.Errorf("tensor[%s] data out of range: offset %d + size %d > file size %d",
				ti.Name, absOffset, byteSize, len(gf.mmapData))
		}
		ti.Data = gf.mmapData[absOffset : absOffset+byteSize]
	}

	return nil
}

// tensorByteSize calculates the payload byte size for a tensor
func tensorByteSize(t GGMLType, dims []uint64) uint64 {
	if len(dims) == 0 {
		return 0
	}
	n := uint64(1)
	for _, d := range dims {
		n *= d
	}

	switch t {
	case GGMLTypeF32:
		return n * 4
	case GGMLTypeF16:
		return n * 2
	case GGMLTypeQ4_0:
		// 32 weights in 18 bytes (2 byte FP16 scale + 16 bytes quants)
		return (n / 32) * 18
	case GGMLTypeQ5_0:
		// 32 weights in 22 bytes (2 byte FP16 scale + 4 bytes high bits + 16 bytes low bits)
		return (n / 32) * 22
	case GGMLTypeQ8_0:
		// 32 weights in 34 bytes (2 byte FP16 scale + 32 bytes int8)
		return (n / 32) * 34
	case GGMLTypeQ4_K:
		// 256 weights in 144 bytes
		return (n / 256) * 144
	case GGMLTypeQ6_K:
		// 256 weights in 210 bytes
		return (n / 256) * 210
	default:
		// Fallback conservative estimate
		return n * 4
	}
}

// GetString returns metadata string
func (gf *File) GetString(key string) (string, bool) {
	v, ok := gf.Metadata[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// GetUint32 returns metadata uint32
func (gf *File) GetUint32(key string) (uint32, bool) {
	v, ok := gf.Metadata[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case uint32:
		return val, true
	case uint64:
		return uint32(val), true
	case int32:
		return uint32(val), true
	case int64:
		return uint32(val), true
	default:
		return 0, false
	}
}

// GetFloat32 returns metadata float32
func (gf *File) GetFloat32(key string) (float32, bool) {
	v, ok := gf.Metadata[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case float32:
		return val, true
	case float64:
		return float32(val), true
	default:
		return 0, false
	}
}

// GetStringSlice returns metadata []string
func (gf *File) GetStringSlice(key string) ([]string, bool) {
	v, ok := gf.Metadata[key]
	if !ok {
		return nil, false
	}
	s, ok := v.([]string)
	return s, ok
}
