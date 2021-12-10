# Redis modules

## Introduction
Redis modules make it possiable to extend Redis functionality using external modules,
repidly implementing new Redis commands features similar to what can be done inside the core itself.

The redis module provided in this directory is dedicated for `BPV2` functionality in TFRA when redis is used as the embedding KV backend, where embedding values could be acumulated
inside redis core.

## Installation
### Build library from source
The module could be compiled from source and requries a minical C build system installed.

```
git clone https://github.com/tensorflow/recommenders-addons.git
cd recommenders-addons/third_party/redis_module/
make
```

The we could get a compiled redis module `*.so`.

### Load the module
We can load the modlue using the following redis.conf configuration derective:
```
loadmodule /path/to/mymodule.so
```
It is also possiable to load a module at runtime using the following command:
```
MODULE LOAD /path/to/mymodule.so
```

## HMACCUM module
In `bpv2_hmaccum_cmd.c` we developed a custome redis command `HMACCUM`, to extend the redis hash table commands(HMSET, HMGET, etc).

```
Hmaccum key valueType field value_or_delta [field value_or_delta...] exists

Return value
Integer reply: the number of fields that were manipulated.
```

The command set the specified fields to their respective values in the hash sotred at key.
If key does not exist, a new key holding a hash is created. If the field already exists in table
and field is marked as exists in exists vector,  the `value_or_delta` will be take as delta and the
value will be accumulated onto the current;
otherwise `value_or_delta` will be take as new value and a new field will be inserted.

The values are stored as continious memory blocks inside redis. The `valueType` parameter is used
to get the underlying value format. And currently onley the following value types supported:

1. float
2. double
3. int32
4. int64
5. int8

### Example

```
127.0.0.1:6379> module load /opt/bpv2_hmaccum_cmd.so
OK
127.0.0.1:6379> HMACCUM myhash "float" "a\xcd\xcf\xc9\x173\x03c" "\x1b\xc5\9c/" "\0"
(integer) 1
127.0.0.1:6379> HMGET myhash "a\xcd\xcf\xc9\x173\x03c"
"\x1b\xc5\x9c/"
127.0.0.1:6379> HMACCUM myhash "float" "a\xcd\xcf\xc9\x173\x03c" "\x1b\xc5\9c/" "\1"
(integer) 1
127.0.0.1:6379> HMGET myhash "a\xcd\xcf\xc9\x173\x03c"
"\x1b\xc5\x1c0"
```
## Reference
[Redis MOdules: an introduction to the API] (https://redis.io/topics/modules-intro)
