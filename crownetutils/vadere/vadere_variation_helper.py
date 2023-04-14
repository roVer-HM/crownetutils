def source_max_spawn(ids, num=0):
    return {f"sources.[id=={id}].maxSpawnNumberTotal": num for id in ids}


def source_spawn(ids, num=0):
    return {f"sources.[id=={id}].spawnNumber": num for id in ids}


def source_dist_par(ids, name, value):
    return {f"sources.[id=={id}].distributionParameters.{name}": value for id in ids}


def source(ids, distribution, params, start, end, spawnNumber, maxSpawnNumber):
    ret = {}
    for id in ids:
        ret[f"sources.[id=={id}].spawnNumber.interSpawnTimeDistribution"] = distribution
        for k, v in params.items():
            ret.update(**source_dist_par([id], k, v))
        ret[f"sources.[id=={id}].spawnNumber.startTime"] = start
        ret[f"sources.[id=={id}].spawnNumber.endTime"] = end
        ret.update(**source_spawn([id], spawnNumber))
        ret.update(**source_max_spawn([id], maxSpawnNumber))
    return ret
