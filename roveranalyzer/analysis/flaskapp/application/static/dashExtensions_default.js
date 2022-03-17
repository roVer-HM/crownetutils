window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        ploy_styler: function(feature, context) {
                return feature.properties['style'];
            }

            ,
        map_style_clb: function(feature, context) {
                const {
                    classes,
                    colorscale,
                    style,
                    colorProp
                } = context.props.hideout; // get props from hideout
                const value = feature.properties[colorProp]; // get value the determines the color
                var ret = classes.findIndex(e => e == value);
                if (ret == -1) {
                    if (value < classes[0]) {
                        ret = 0;
                    } else {
                        ret = classes[classes.length - 1];
                    }
                }
                style.fillColor = colorscale[ret];
                style.color = colorscale[ret];
                return style;
            }

            ,
        node_point_to_layer: function(feature, latlng, context) {
            const {
                circleOptions,
                colorProp
            } = context.props.hideout;
            //const csc = chroma.scale(colorscale).domain([min, max]);  // chroma lib to construct colorscale
            circleOptions.fillColor = feature.properties[colorProp] //csc(feature.properties[colorProp]);  // set color based on color prop.
            return L.circleMarker(latlng, circleOptions); // sender a simple circle marker.
        }

    }
});