====================================
Arch Linux
====================================

:著者: Masato

概要
====================================
Linuxのディストリビューションの一つであるArch Linuxについてインストールに必要なものから、インストール方法、
実際に使っているアプリケーションにいたるまで、やってきたことをまとめます。

インストール
====================================

パーティション
------------------------------------
UEFIの場合は/bootは512MB以上のvfatが必要。今回の構成は/bootが512MB以上で、それ以外は全て/rootでext4とした。
まずは、新規のディスクなので、oをタイプし、guidパーティションテーブルを作成する。

次いでn(add a new partition)をタイプする。ここでは、

* Partition number 1 に 512MB
* Typeは、ef00 EFI System

を選択する。再びnをタイプし、

* Partition number 2 に残り全部
* Typeは8e00 Linux LVMを選択?

今回swapは作成しないが、必要に応じて行う。

ファイルシステムの構築
------------------------------------
それぞれのパーティションをフォーマットする。 ::

    # mkfs.vfat -F32 /dev/sda1
    # mkfs.ext4 /dev/sda2  

ベースシステムのインストール
------------------------------------
ファイルシステムのマウントを先に行う。::

    # mount /dev/sda2 /mnt
    # mkdir -p /mnt/boot
    # mount /dev/sda1 /mnt/boot

インストールするためには、まず、::

    # vi /etc/pacman.d/mirrorlist

で、::

    Server = http://ftp.jaist.ac.jp/pub/Linux/ArchLinux/$repo/os/$arch

をServerのリストの先頭に配置する。その上で、::

    # pacstrap - i /mnt base base-devel

とする。インストール対象パッケージを選別したい場合にには、-iオプションを外す必要がある。

システム設定
------------------------------------
fstabを生成するために、::

    # gengstab -U -p /mnt >> /mnt/etc/fstab

ルートをマウント先に変更する。::

    # arch-chroot /mnt

Localeを設定するために、::

    # vi /etc/locale.gen

にて、以下の2行のコメントアウトを外す。::

    en_US.UTF-8 UTF-8
    ja_JP.UTF-8 UTF-8

書き換えた終わったら、::

    # locale-gen
    # echo LANG=ja_JP.UTF-8 > /etc/locale.conf

次にタイムゾーン、ハードウェアクロックの設定である。::

    ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
    hwclock --systohc --utc

マシンのホスト名を決める。::

    # echo 名前 > /etc/hostname

initial ramdisk環境の作成
------------------------------------
これは、::

    # mkinitcpio -p linux

これが終わったら、rootのパスワード設定も同時に行う。::

    # passed
    
grubなどの設定を行う。::

    # pacman -S grub efibootmgr
    # grum-install --target='i386-efi' /dev/sda

設定が終わったら、アンマウントとリブートを行う。::

    # exit
    # umount /mnt/boot
    # umount /mnt
    # reboot

マシンの設定
------------------------------------
起動したら、rootでログインする。最初の状態では、LANGにja_JP.UTF-8が指定されていることがあるため、Xが起動していないと、コンソールの日本語が化ける。
そのため、::

    # export LANG=en_UT.UTF-8

により、ASCIIだけにしておく。

パッケージの最新化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
先にソフトウェアの更新を行う。::

    # pacman -Syu


個人ユーザの作成
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
rootでうろうろするのは危険なので、個人ユーザを作成する。::

    # useradd -m {username}
    # passwd {username}
    # pacman -S sudo
    # visudo
    {username} ALL=(ALL) ALL        <-個人ユーザがsudo出きるようにこれを記述

X導入
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ある程度GUIで設定したほうが楽なので、Xを導入する。::

    # pacman -S xorg-server xterm xorg-server-utils xorg-xinit xorg-xclock 

ドライバを導入すために、::

    # lspci | grep VGA

とすると、::

    00:02.0 VGA compatible controller: Intel Corporation Atom Processor Z36xxx/Z37xxx Series Graphics & Display (rev 0f)

となっているおり、intelがあるので、::

    # pacman -S xf8-video-intel

とする。

コマンド導入
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
最低限使いやすいようにいくつかのコマンドを先に入れる。::

    # pacman -S zsh tmux openssl git vim yaourt

zshに変更するには、::

    # chsh

で、シェルの場所を教える。

Yaourtの導入に伴い、設定をする。::

    # vim /etc/pacman.conf

    以下を追加
    [archlinuxfr]
    SigLevel = Never
    Server = http://repo.archlinux.fr/$arch

    [pnsft-pur]
    SigLevel = Optional TrustAll
    Server = http://downloads.sourceforge.net/project/pnsft-aur/pur/$arch

    multilibのコメントアウトを外して、32bitパッケージを利用可能にする。::
    [multilib]
    Include = /etc/pacman.d/mirrorlist

    以上が終わったら、リフレッシュする。
    # pacman --sync --refresh yaourt
    # pacman -Syu

GUI画面変更
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SLiMによって、startxなしで起動時に自動的にXが起動するようにする。::

    # pacman -S slim archlinx-themes-slim slim-themes
    # vi /etc/slim.conf

以下のように編集とコメントアウトを外す。::
    login_cmd exec /bin/zsh -l ~/.xinitrc %session
    daemon yes
    current_theme archlinux-simplyblack

最後に、::

    # systemctl enable slim.service

この時にログイン先のセッションを指定する必要がある。WMの導入をする。::

    # pacman -S xfce4 xfce4-goodies gamin
    # cp /etc/skel/.xinitrc ~/
    # vi ~/.xinitrc

以下のコメントアウトを外す。::

    exec startxfce4

これでrootのGUIはだいたい揃った。

日本語化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
日本語フォントとmozcを導入する。::

    # yaourt -S ttf-ricty
    # pacman -S mozc ibus-mozc

    ~/.xinitrcに以下を追加

    export GTK_IM_MODULE=ibus
    export XMODIFIERS=@im=ibus
    export QT_IM_MODULE=ibus
    ibus-daemon  --xim -d &

ここで、ibusの設定をする。::

    $ ibus-setup
    ※pythonを使用しているので、この段階でAnacondaを入れている場合は、デフォルトのpythonが動くように環境変数を戻す。

GUIまとも化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
壁紙を追加する。壁紙の設定をするのはnitrogenで、::
    
    # pacman -S nitrogen
    $ nitrogen {wallpapers_path}

引数はディレクトリなので、注意。その上で、~/.xinitrcに::

    nitrogen --restore &

を記述する。xmonadについては後述する予定。

便利化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
外部メディアやディスクの自動マウント設定::

    # pacman -S gvfs

Xtermの設定
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
xterm起動時は基本的にに白い画面になってしまうため、Xtermの設定が必要である。
Xtermの設定は.Xresourcesに書く。 ::

    XTerm*utf8              : 1
    XTerm*locale            : true
    XTerm*selectToClipboard : true
    XTerm*faceName          : 'Monospace'
    XTerm*faceSize          : 13
    XTerm*cjkWidth          : true
    XTerm*background        : black
    XTerm*foreground        : white
    XTerm*saveLines         : 2000
    XTerm*geometry          : 100x30+300+100

最低限必要なのは、faceNameとfacesizeであり、それぞれ、使用フォントとサイズである。
使えるフォントを調べるには、 ::
    $ fc-list

あとで細かく追記する。
http://note.kurodigi.com/xterm-customize/
http://incompleteness-theorems.at.webry.info/201009/article_6.html
その後、ターミナルより、 ::

    $ xrdb .Xresouces

を実行して、反映する。これで、次回の実行時から設定が反映される。

起動しないときに
====================================
カーネルアップデート後に起動しなくなったときに、以下の手順でカーネルの再構築を試して見てください。
起動しなくなったLinuxが存在するrootパーティションをマウントする。例えば、/mnt/archにマウントする。::

    $ mkdir /mnt/arch
    $ mount /dev/sda1 /mnt/arch

/bootや/varが別のパーティションがある場合は、マウントする。例えば、::
    
    $ mount /dev/sda2 /mnt/arch/boot

など。
さらに、APIファイルシステムをマウントする。::

    $ cd /mnt/arch
    $ mount -t proc proc proc/
    $ mount --rbind /sys sys/
    $ mount --rbind /dev dev/

ファイルシステムのマウントができたら、chrootする。::

    $ chroot /mnt/arch /bin/bash

必要であれば、ネットの情報をコピーする。※追記

chrootができたら、カーネルイメージを作り直す。イメージを作る前に、udevとmkinitcpioを再インストールする。::

    $ pacman -Syy
    $ pacman -Syu
    $ pacman -S udev
    $ pacman -S mkinitcpio
    $ pacman -S linux
    $ mkinitcpio -p linux

カーネルイメージの再構成が成功したら、chrootを抜けだし、再起動する。::

    $ exit
    $ cd /
    $ umount --recursive /mnt/arch/
    $ reboot

これで大抵OSが起動するようになる。Arch Linuxが起動しなくなるのは、アップデート時にカーネルイメージがうまく再構成できなかったり、
udevやmkinitcpioが上手くアップデートできないときに起こる。
