var data = []
var token = ""

jQuery(document).ready(function () {
    var slider_sentences = $('#max_sentences')
    slider_sentences.on('change mousemove', function (evt) {
        $('#label_max_sentences').text('sentences: ' + slider_sentences.val())
    })

    var slider_beamsearch = $('#beam_search')
    slider_beamsearch.on('change mousemove', function (evt) {
        $('#label_beam_searcher').text('beam search: ' + slider_beamsearch.val())
    })


    $(document).on('click', '#btn_generate', function (e) {
        $.ajax({
            url: '/get_paraphrase',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_text": $('#input_text').val(),
                "num_sentences": slider_sentences.val(),
                "beam_search": $('#beam_search').val(),
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $('#text_t5').val(jsondata)
        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
        });
    })

})