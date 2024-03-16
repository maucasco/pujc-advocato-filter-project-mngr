import exifread

def obtener_informacion_camara(imagen_path):
    properties={}
    with open(imagen_path, 'rb') as imagen:
        exifreadValues = exifread.process_file(imagen)
        tags = {}

        # Iterar sobre las etiquetas EXIF y eliminar las que contienen valores ASCII
        for tag, valor in exifreadValues.items():
            if isinstance(valor, bytes):
                try:
                    valor = valor.decode('utf-8')
                except UnicodeDecodeError:
                    pass
            tags[tag] = valor
        properties={
            "make":tags.get('Image Make', 'No disponible'),
            "model" : tags.get('Image Model', 'No disponible'),
            "latitude" : tags.get('GPS GPSLatitudeRef', 'No disponible'),
            "longitude" : tags.get('GPS GPSLongitude', 'No disponible'),
            "ImageDateTime" : tags.get('Image DateTime', 'No disponible'),
            "Image YCbCrPositioning" : tags.get('ImageYCbCrPositioning', 'No disponible'),
            "ImageExifOffset" : tags.get('Image ExifOffset', 'No disponible'),
            "ImageResolutionUnit" : tags.get('Image ResolutionUnit', 'No disponible'),
            "GPSLatitudeRef" : tags.get('GPS GPSLatitudeRef', 'No disponible'),
            "GPSLatitude" : tags.get('GPS GPSLatitude', 'No disponible'),

            "GPSLongitudeRef" : tags.get('GPS GPSLongitudeRef', 'No disponible'),
            "GPSLongitude" : tags.get('GPS GPSLongitude', 'No disponible'),

            "GPSAltitudeRef" : tags.get('GPS GPSAltitudeRef', 'No disponible'),
            "GPSAltitude" : tags.get('GPS GPSAltitude', 'No disponible'),
            "GPSTimeStamp" : tags.get('GPS GPSTimeStamp', 'No disponible'),
            "GPSProcessingMethod" : tags.get('GPS GPSProcessingMethod', 'No disponible'),
            "GPSDate" : tags.get('GPS GPSDate', 'No disponible'),

            "ImageGPSInfo" : tags.get('Image GPSInfo', 'No disponible'),
            "ImageXResolution" : tags.get('Image XResolution', 'No disponible'),
            "ImageYResolution" : tags.get('Image YResolution', 'No disponible'),

            "EXIFSubSecTimeDigitized":tags.get('EXIF SubSecTimeDigitized', 'No disponible'),
            "EXIFSubSecTimeOriginal":tags.get('EXIF SubSecTimeOriginal', 'No disponible'),
            "EXIFSubSecTime":tags.get('EXIF SubSecTime', 'No disponible'),
            "EXIFFocalLength":tags.get('EXIF FocalLength', 'No disponible'),
            "EXIFFlash":tags.get('EXIF Flash', 'No disponible'),
            "EXIFMeteringMode":tags.get('EXIF MeteringMode', 'No disponible'),
            "EXIFSceneCaptureType":tags.get('EXIF SceneCaptureType', 'No disponible'),
            "InteroperabilityIndex":tags.get('Interoperability InteroperabilityIndex', 'No disponible'),
            "InteroperabilityVersion":tags.get('Interoperability InteroperabilityVersione', 'No disponible'),

            "EXIFISOSpeedRatings" : tags.get('EXIF ISOSpeedRatings', 'No disponible'),
            "EXIFExposureProgram" : tags.get('EXIF ExposureProgram', 'No disponible'),
            "EXIFFNumber" : tags.get('EXIF FNumber', 'No disponible'),
            "EXIFExposureTime" : tags.get('EXIF ExposureTime', 'No disponible'),
            "EXIFSensingMethod" : tags.get('EXIF SensingMethod', 'No disponible'),
            "EXIFInteroperabilityOffset": tags.get('EXIF InteroperabilityOffset', 'No disponible'),
            "EXIFFocalLengthIn35mmFilm": tags.get('EXIF FocalLengthIn35mmFilm', 'No disponible'),
            "EXIFDateTimeDigitized": tags.get('EXIF DateTimeDigitized', 'No disponible'),
            "EXIFExifImageLength": tags.get('EXIF ExifImageLength', 'No disponible'),
            "EXIFWhiteBalance": tags.get('EXIF WhiteBalance"', 'No disponible'),
            "EXIFDateTimeOriginal": tags.get('EXIF DateTimeOriginal', 'No disponible'),
            "EXIFBrightnessValue": tags.get('EXIF BrightnessValue', 'No disponible'),
            "EXIFExifImageWidth": tags.get('EXIF ExifImageWidth', 'No disponible'),
            "EXIFExposureMode": tags.get('EXIF ExposureMode', 'No disponible'),
            "EXIFApertureValue": tags.get('EXIF ApertureValue', 'No disponible'),
            "EXIFComponentsConfiguration": tags.get('EXIF ComponentsConfiguration', 'No disponible'),
            "EXIFColorSpace": tags.get('EXIF ColorSpace', 'No disponible'),
            "EXIFSceneType": tags.get('EXIF SceneType', 'No disponible'),
            "EXIFShutterSpeedValue": tags.get('EXIF ShutterSpeedValue', 'No disponible'),
            "EXIFExifVersion": tags.get('EXIF ExifVersion', 'No disponible'),
            "EXIFFlashPixVersion": tags.get('EXIF FlashPixVersion', 'No disponible'),
        }
    lat,lot= get_exif_location(tags)
    properties['latitude']=lat
    properties['longitude']=lot
    print(properties)
    return properties

def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None

def _convert_to_degress(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)

    return d + (m / 60.0) + (s / 3600.0)
    
def get_exif_location( exif_data):
    """
    Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)
    """
    lat = None
    lon = None

    gps_latitude =_get_if_exist(exif_data, 'GPS GPSLatitude')
    gps_latitude_ref = _get_if_exist(exif_data, 'GPS GPSLatitudeRef')
    gps_longitude = _get_if_exist(exif_data, 'GPS GPSLongitude')
    gps_longitude_ref =_get_if_exist(exif_data, 'GPS GPSLongitudeRef')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = _convert_to_degress(gps_latitude)
        if gps_latitude_ref.values[0] != 'N':
            lat = 0 - lat

        lon = _convert_to_degress(gps_longitude)
        if gps_longitude_ref.values[0] != 'E':
            lon = 0 - lon

    return lat, lon

