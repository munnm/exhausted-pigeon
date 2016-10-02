def recommendation(a,b,c,d,e,f,g,h,i):
    sentence = "With %r%% accuracy, you should get at least %r hours of sleep a night \n" \
        " and you do best when you're in bed by %r:%r:%r. \n" \
        "Over the last two weeks, you generally got about %r hours of sleep each night" \
        " and you were in bed by %r:%r:%r." % (a,b,c,d,e,f,g,h,i)
    return sentence
